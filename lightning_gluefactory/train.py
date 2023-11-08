import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Tuple

import aim
import hydra
import numpy as np
import torch
from aim import Figure, Image
from hydra.core.config_store import ConfigStore
from lightning.pytorch.loggers import TensorBoardLogger
from aim.pytorch_lightning import AimLogger
from omegaconf import OmegaConf, DictConfig, MISSING, ListConfig
from torchmetrics import MeanMetric
from pydantic import BaseModel as PydanticBaseModel

from gluefactory.train import filter_parameters, pack_lr_parameters
from gluefactory.utils.tools import set_seed, PRMetric
from lightning_gluefactory import __module_name__, logger
from lightning_gluefactory.datasets.homographies import HomographyDataset
from lightning_gluefactory.datasets.base_dataset import BaseDataModule, BaseDatasetConfig

import lightning as L

from lightning_gluefactory.models.base_model import BaseModel, BaseModelConfig


class OptimizerConfig(PydanticBaseModel):
    """
    Configuration for the optimizer.
    """
    _target_: str = "torch.optim.Adam"


class LRScheduleConfig(PydanticBaseModel):
    """
    Configuration for the learning rate schedule.
    """
    type: Any = None
    start: int = 0
    exp_div_10: int = 0
    on_epoch: bool = False
    factor: float = 1.0


class TrainConfig:
    """
    Configuration for the training.
    """

    seed: int
    epochs: int = 1
    opt_regexp: Optional[str] = None
    optimizer: OptimizerConfig = OptimizerConfig()
    optimizer_options: dict = field(default_factory=dict)
    lr: float = 0.001
    lr_schedule: LRScheduleConfig = LRScheduleConfig()
    lr_scaling: List[Tuple[int, str]] = [(100, "dampingnet.const")]
    eval_every_iter: int = 1000
    save_every_iter: int = 5000
    log_every_iter: int = 100
    log_grad_every_iter: Optional[int] = None
    test_every_epoch: int = 1
    keep_last_checkpoints: int = 10
    load_experiment: Optional[str] = None
    median_metrics: List[str] = []
    recall_metrics: dict = {}
    pr_metrics: dict = {}
    best_key: str = "loss/total"
    dataset_callback_fn: Optional[str] = None
    dataset_callback_on_val: Optional[bool] = False
    clip_grad: Optional[float] = None
    pr_curves: dict = {}
    plot: bool = False
    submodules: List[str] = []
    strategy: str = "auto"
    train_metric_prefix: str = "train/"
    val_metric_prefix: str = "val/"


class GlueFactoryConfig(PydanticBaseModel):
    train: TrainConfig
    data: BaseDatasetConfig
    model: BaseModelConfig
    callbacks: dict = {}

    class Config:
        arbitrary_types_allowed = True



class GlueFactory(L.LightningModule):
    def __init__(self, config: GlueFactoryConfig):
        super().__init__()
        self.config: GlueFactoryConfig = config

        self.model: BaseModel = hydra.utils.instantiate(config.model)
        self.loss_fn = self.model.loss
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)

        if self.config.get("compile", False):
            self.model = torch.compile(self.model, mode=self.config.compile)

        self.validation_plot = False
        if self.config.train.get("plot", False):
            self.validation_plot = True
            self.val_number_of_plots, self.val_plot_function = self.config.train.plot
        self.mean_metric = MeanMetric(nan_strategy="ignore")

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        L.seed_everything(self.config.train.seed + self.current_epoch)

    def training_step(self, batch, batch_idx):
        data = batch
        pred = self.model(data)
        losses, _ = self.loss_fn(pred, data)
        loss = torch.mean(losses["total"])
        self.log(
            self.config.train.train_metric_prefix + "loss/total",
            loss,
            rank_zero_only=True,
        )
        if torch.isnan(loss).any():
            logger.log(f"Detected NAN, skipping iteration {batch_idx}")
            del pred, data, loss, losses
        else:
            return loss

    def on_validation_epoch_start(self) -> None:
        self.val_results = {}
        # self.pr_metrics = defaultdict(PRMetric)
        if self.validation_plot:
            self.val_figures = []
            self.val_batch_ids_to_plot = np.random.choice(
                self.config.data.val_size,
                min(self.config.data.val_size, self.val_number_of_plots),
                replace=False,
            )

    def validation_step(self, batch, batch_idx):
        data = batch
        pred = self.model(data)
        losses, metrics = self.loss_fn(pred, data)
        if self.validation_plot and batch_idx in self.val_batch_ids_to_plot:
            validation_figure = hydra.utils.call(self.val_plot_function, pred, data)
            for k, v in validation_figure.items():
                self.logger.experiment.track(Image(v), name=k, step=batch_idx)
            # TODO pr_curves implementation
        numbers = {
            **metrics,
            **{
                self.config.train.val_metric_prefix + "loss/" + k: v
                for k, v in losses.items()
            },
        }

        for k, v in numbers.items():
            if k not in self.val_results:
                self.val_results[k] = MeanMetric(nan_strategy="ignore").to(v)
            self.val_results[k].update(v)
        # TODO implement median and recall metrics
        # if k in self.config.median_metrics:
        #     self.results[k + "_median"] = MedianMetric()
        # if k in self.config.recall_metrics.keys():
        #     q = self.config.recall_metrics[k]
        #     self.results[k + f"_recall{int(q)}"] = RecallMetric(q)
        # if k in self.config.median_metrics:
        #     self.results[k + "_median"].update(v)
        # if k in self.config.recall_metrics.keys():
        #     q = self.config.recall_metrics[k]
        #     self.results[k + f"_recall{int(q)}"].update(v)
        return numbers

    def on_validation_epoch_end(self) -> None:
        self.val_results = {k: self.val_results[k].compute() for k in self.val_results}
        for k, v in self.val_results.items():
            self.log(k, v, rank_zero_only=True, sync_dist=True)

    def configure_optimizers(self):
        # optimizer_fn = hydra.utils.call(self.config.train.optimizer, params=self.model.parameters())
        params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if self.config.train.opt_regexp:
            params = filter_parameters(params, self.config.train.opt_regexp)
        # all_params = [p for n, p in params]
        lr_params = pack_lr_parameters(
            params, self.config.train.lr, self.config.train.lr_scaling
        )
        optimizer = hydra.utils.instantiate(
            self.config.train.optimizer,
            lr_params,
            lr=self.config.train.lr,
            **self.config.train.optimizer_options,
        )

        def lr_fn(it):  # noqa: E306
            if self.config.train.lr_schedule.type is None:
                return 1
            if self.config.train.lr_schedule.type == "factor":
                return (
                    1.0
                    if it < self.config.train.lr_schedule.start
                    else self.config.train.lr_schedule.factor
                )
            if self.config.train.lr_schedule.type == "exp":
                gam = 10 ** (-1 / self.config.train.lr_schedule.exp_div_10)
                return 1.0 if it < self.config.train.lr_schedule.start else gam
            else:
                raise ValueError(self.config.train.lr_schedule.type)

        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def training(config: GlueFactoryConfig):
    # TODO implement restore

    # TODO implement load experiment

    set_seed(config.train.seed)

    datamodule: BaseDataModule = hydra.utils.instantiate(config.data)

    # print(datamodule)

    # datamodule.prepare_data()
    # datamodule.setup("fit")

    # train_dataloader = datamodule.train_dataloader()
    # print(train_dataloader)
    # val_dataloader = datamodule.val_dataloader()
    # print(val_dataloader)

    # TODO implement validation dataset being different from training one

    # TODO implement overfit mode

    # logger.info(f"Training loader has {len(train_dataloader)} batches")
    # logger.info(f"Validation loader has {len(val_dataloader)} batches")

    # for batch in train_dataloader:
    # print(batch)
    # break

    # model : BaseModel = hydra.utils.instantiate(config.model)
    # print(model)

    # TODO implement training restoration

    # TODO implement profiler

    # TODO run benchmarks

    glue_factory = GlueFactory(config)

    logger: AimLogger = (
        hydra.utils.instantiate(config.logger)
        if getattr(config, "logger", None) is not None
        else None
    )
    logger.log_hyperparams(config)
    callbacks = [hydra.utils.instantiate(c) for c in config.callbacks.values()]

    print(os.getcwd())
    trainer = L.Trainer(
        max_epochs=config.train.epochs,
        log_every_n_steps=config.train.log_every_iter,
        logger=logger,
        callbacks=callbacks,
        # strategy=config.train.strategy,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(glue_factory, datamodule)


@hydra.main(config_path="configs", config_name="train")
def main(config: GlueFactoryConfig):
    print(OmegaConf.to_yaml(config))
    OmegaConf.resolve(config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    glue_factory_config = GlueFactoryConfig(*config_dict)
    training_path = Path(config.training_path)
    training_path.mkdir(parents=True, exist_ok=True)

    training(glue_factory_config)


if __name__ == "__main__":
    main()
