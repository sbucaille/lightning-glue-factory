"""
Base class for dataset.
See mnist.py for an example of dataset.
"""

import collections
import logging
from abc import ABCMeta, abstractmethod

import lightning as L
import omegaconf
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler, get_worker_info
from torch.utils.data._utils.collate import (
    default_collate_err_msg_format,
    np_str_obj_array_pattern,
)

from gluefactory.utils.tensor import string_classes
from gluefactory.utils.tools import set_num_threads, set_seed

logger = logging.getLogger(__name__)


class LoopSampler(Sampler):
    def __init__(self, loop_size, total_size=None):
        self.loop_size = loop_size
        self.total_size = total_size - (total_size % loop_size)

    def __iter__(self):
        return (i % self.loop_size for i in range(self.total_size))

    def __len__(self):
        return self.total_size


def worker_init_fn(i):
    info = get_worker_info()
    if hasattr(info.dataset, "conf"):
        conf = info.dataset.conf
        set_seed(info.id + conf.seed)
        set_num_threads(conf.num_threads)
    else:
        set_num_threads(1)


def collate(batch):
    """Difference with PyTorch default_collate: it can stack of other objects."""
    if not isinstance(batch, list):  # no batching
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            try:
                storage = elem.untyped_storage()._new_shared(numel)  # noqa: F841
            except AttributeError:
                storage = elem.storage()._new_shared(numel)  # noqa: F841
        return torch.stack(batch, dim=0)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif elem is None:
        return elem
    else:
        # try to stack anyway in case the object implements stacking.
        return torch.stack(batch, 0)


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle_training: bool,
        batch_size: int,
        num_threads: int,
        seed: int,
        prefetch_factor: int,
    ):
        super().__init__()
        """Perform some logic and call the _init method of the child model."""
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle_training = shuffle_training
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.seed = seed
        self.prefetch_factor = prefetch_factor

        logger.info(f"Creating dataset {self.__class__.__name__}")

    def prepare_data(self) -> None:
        raise NotImplementedError

    def setup(self, stage: str) -> None:
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader("train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader("val")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader("test")

    def get_dataloader(self, split: str):
        assert split in ["train", "val", "test"]
        dataset = self.__getattribute__(split + "_dataset")
        sampler = LoopSampler(
            self.batch_size,
            len(dataset) if split == "train" else self.batch_size,
        )
        num_workers = self.num_workers
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            sampler=sampler,
            # worker_init_fn=worker_init_fn,
            persistent_workers=True,
            collate_fn=collate,
        )


class BaseDataset(metaclass=ABCMeta):
    """
    What the dataset model is expect to declare:
        default_conf: dictionary of the default configuration of the dataset.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.

        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unknown configuration entries will raise an error.

        get_dataset(self, split): method that returns an instance of
        torch.utils.data.Dataset corresponding to the requested split string,
        which can be `'train'`, `'val'`, or `'test'`.
    """

    base_default_conf = {
        "name": "???",
        "num_workers": "???",
        "train_batch_size": "???",
        "val_batch_size": "???",
        "test_batch_size": "???",
        "shuffle_training": True,
        "batch_size": 1,
        "num_threads": 1,
        "seed": 0,
        "prefetch_factor": 2,
    }
    default_conf = {}

    def __init__(
        self,
        num_workers: int,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle_training: bool,
        batch_size: int,
        num_threads: int,
        seed: int,
        prefetch_factor: int,
    ):
        """Perform some logic and call the _init method of the child model."""
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle_training = shuffle_training
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.seed = seed
        self.prefetch_factor = prefetch_factor

        logger.info(f"Creating dataset {self.__class__.__name__}")

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, split):
        """To be implemented by the child class."""
        raise NotImplementedError

    def get_data_loader(self, split, shuffle=None, pinned=False, distributed=False):
        """Return a data loader for a given split."""
        assert split in ["train", "val", "test"]
        dataset = self.get_dataset(split)
        batch_size = self.__getattribute__(split + "_batch_size")
        num_workers = self.num_workers

        sampler = None
        if shuffle is None:
            shuffle = split == "train" and self.shuffle_training
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=pinned,
            collate_fn=collate,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.prefetch_factor,
            drop_last=True if split == "train" else False,
        )

    def get_overfit_loader(self, split):
        """Return an overfit data loader.
        The training set is composed of a single duplicated batch, while
        the validation and test sets contain a single copy of this same batch.
        This is useful to debug a model and make sure that losses and metrics
        correlate well.
        """
        assert split in ["train", "val", "test"]
        dataset = self.get_dataset("train")
        sampler = LoopSampler(
            self.batch_size,
            len(dataset) if split == "train" else self.batch_size,
        )
        num_workers = self.num_workers
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=num_workers,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
