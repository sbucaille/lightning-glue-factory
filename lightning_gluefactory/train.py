from pathlib import Path

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger

from gluefactory.utils.tools import set_seed
from lightning_gluefactory.datasets.base_dataset import BaseDataModule
from lightning_gluefactory.glue_factory import GlueFactory


def training(config: DictConfig):
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

    logger: Logger = hydra.utils.instantiate(config.logger) if getattr(config, "logger", None) is not None else None
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
    OmegaConf.resolve(config)
    print(OmegaConf.to_yaml(config))

    training_path = Path(config.training_path)
    training_path.mkdir(parents=True, exist_ok=True)

    training(config)


if __name__ == "__main__":
    main()
