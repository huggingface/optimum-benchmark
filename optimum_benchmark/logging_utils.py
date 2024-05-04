import logging
import logging.config
from logging import Logger
from subprocess import PIPE, STDOUT, Popen
from typing import Optional


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        self.logs.append(self.format(record))

    def get_logs(self):
        return self.logs


LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {"format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"},
        "colorlog": {
            "()": "colorlog.ColoredFormatter",
            "format": "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            "log_colors": {"DEBUG": "purple", "INFO": "green", "WARNING": "yellow", "CRITICAL": "red", "ERROR": "red"},
        },
    },
    "handlers": {
        "console": {"formatter": "colorlog", "stream": "ext://sys.stdout", "class": "logging.StreamHandler"},
        "file": {"formatter": "simple", "filename": "benchmark.log", "class": "logging.FileHandler"},
        "colored_file": {"formatter": "colorlog", "filename": "benchmark.log", "class": "logging.FileHandler"},
        "list": {"formatter": "simple", "class": "optimum_benchmark.logging_utils.ListHandler"},
        "colored_list": {"formatter": "colorlog", "class": "optimum_benchmark.logging_utils.ListHandler"},
    },
    "root": {"level": "INFO", "handlers": ["console", "file", "colored_file", "list", "colored_list"]},
    "disable_existing_loggers": False,
}


def get_logs(colored: bool = False):
    """
    Returns the logs of the benchmark
    """

    for handler in logging.getLogger().handlers:
        if isinstance(handler, ListHandler) and colored and "colored" in handler.get_name():
            return handler.get_logs()
        elif isinstance(handler, ListHandler) and not colored and "colored" not in handler.get_name():
            return handler.get_logs()

    return []


def setup_logging(level: str = "INFO", prefix: Optional[str] = None):
    logging_config = LOGGING_CONFIG.copy()
    logging_config["root"]["level"] = level

    if prefix is not None:
        logging_config["formatters"]["simple"]["format"] = (
            f"[{prefix}]" + logging_config["formatters"]["simple"]["format"]
        )
        logging_config["formatters"]["colorlog"]["format"] = (
            f"[{prefix}]" + logging_config["formatters"]["colorlog"]["format"]
        )

    logging.config.dictConfig(logging_config)


def run_subprocess_and_log_stream_output(logger: Logger, args):
    popen = Popen(args, stdout=PIPE, stderr=STDOUT)
    for line in iter(popen.stdout.readline, b""):
        if line is not None:
            logger.info(line.decode("utf-8").rstrip())

    popen.wait()
    return popen
