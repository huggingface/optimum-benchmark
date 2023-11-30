from logging import getLogger
from typing import Callable

from ..base import Launcher
from .config import InlineConfig

LOGGER = getLogger("inline")


class InlineLauncher(Launcher[InlineConfig]):
    NAME = "inline"

    def __init__(self) -> None:
        super().__init__()

    def configure(self, config: InlineConfig) -> None:
        super().configure(config)

    def launch(self, worker: Callable, *worker_args):
        worker(*worker_args)
