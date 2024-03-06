from typing import Any

from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks.callback import Callback
from time import time

from lightning.pytorch.profilers import Profiler
from lightning.pytorch.utilities.types import STEP_OUTPUT


class SpeedMonitorCallback(Callback):

    def __init__(self, log_every_iter: int):
        self.log_every_iter = log_every_iter
        self.epoch_timer = None
        self.batch_timer = None
        self.batch_idx = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.epoch_timer = time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        now = time()
        elapsed_time = now - self.epoch_timer
        trainer.logger.log_metrics({"epoch_time_elapsed": elapsed_time}, step=trainer.global_step)
        self.epoch_timer = None

    def on_train_batch_start(
            self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self.batch_timer = time()
        self.batch_idx = batch_idx

    def on_train_batch_end(
            self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        now = time()
        elapsed_time = now - self.batch_timer
        batch_per_seconds = self.log_every_iter / elapsed_time
        trainer.logger.log_metrics({"batch_per_seconds": batch_per_seconds}, step=trainer.global_step)
        self.batch_idx = None
        self.batch_timer = None
