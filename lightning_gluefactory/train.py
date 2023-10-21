import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from gluefactory.utils.tools import set_seed
from lightning_gluefactory import __module_name__, logger
from lightning_gluefactory.datasets.homographies import HomographyDataset
from lightning_gluefactory.datasets.base_dataset import BaseDataModule

import lightning as L

from lightning_gluefactory.models.base_model import BaseModel


def training(config: OmegaConf):
	# TODO implement restore

	# TODO implement load experiment

	set_seed(config.train.seed)

	# datamodule: BaseDataModule = hydra.utils.instantiate(config.data)

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

	model : BaseModel = hydra.utils.instantiate(config.model)
	print(model)


@hydra.main(config_path="configs", config_name="train")
def main(config: OmegaConf):
	print(OmegaConf.to_yaml(config))

	training_path = Path(config.training_path)
	training_path.mkdir(parents=True, exist_ok=True)

	training(config)


if __name__ == "__main__":
	main()
