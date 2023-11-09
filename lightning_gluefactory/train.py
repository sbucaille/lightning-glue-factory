import argparse
import os
from collections import defaultdict
from pathlib import Path

import aim
import hydra
import numpy as np
import torch
from aim import Figure, Image
from lightning.pytorch.loggers import TensorBoardLogger
from aim.pytorch_lightning import AimLogger
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import Logger
from torchmetrics import MeanMetric

from gluefactory.train import filter_parameters, pack_lr_parameters
from gluefactory.utils.tools import set_seed, PRMetric
from lightning_gluefactory import __module_name__, logger
from lightning_gluefactory.datasets.homographies import HomographyDataset
from lightning_gluefactory.datasets.base_dataset import BaseDataModule

import lightning as L

from lightning_gluefactory.models.base_model import BaseModel


class GlueFactory(L.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config: DictConfig = config

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
        if torch.isnan(losses["total"]).any():
            del pred, data, losses
            return None
        for k, v in losses.items():
            losses[k] = torch.mean(v)
            self.log(
                self.config.train.train_metric_prefix + "loss/" + k,
                losses[k],
                rank_zero_only=True,
            )
        losses["loss"] = losses["total"]
        return losses

    def on_validation_epoch_start(self) -> None:
        self.val_results = {}
        # self.pr_metrics = defaultdict(PRMetric)
        if self.validation_plot:
            self.val_figures = []
            self.val_batch_ids_to_plot = np.random.choice(
                self.config.data.val_size // self.config.data.batch_size,
                min(self.config.data.val_size // self.config.data.batch_size, self.val_number_of_plots),
                replace=False,
            )

    def validation_step(self, batch, batch_idx):
        data = batch
        pred = self.model(data)
        losses, metrics = self.loss_fn(pred, data)
        if self.validation_plot and batch_idx in self.val_batch_ids_to_plot:
            validation_figure = hydra.utils.call(self.val_plot_function, pred, data)
            for k, v in validation_figure.items():
                self.logger.experiment.track(Image(v), name=k, step=batch_idx, context={'subset': 'val'})
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


def training(config: OmegaConf):
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

    logger: Logger = (
        hydra.utils.instantiate(config.logger)
        if getattr(config, "logger", None) is not None
        else None
    )
    logger.log_hyperparams(config)
    callbacks = [hydra.utils.instantiate(c) for c in config.callbacks.values()]

    trainer = L.Trainer(
        max_epochs=config.train.epochs,
        log_every_n_steps=config.train.log_every_iter,
        val_check_interval=config.train.eval_every_iter,
        logger=logger,
        callbacks=callbacks,
        # strategy=config.train.strategy,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(glue_factory, datamodule)


@hydra.main(config_path="configs", config_name="train")
def main(config: OmegaConf):
    print(OmegaConf.to_yaml(config))

    training_path = Path(config.training_path)
    training_path.mkdir(parents=True, exist_ok=True)

    training(config)


if __name__ == "__main__":
    main()
