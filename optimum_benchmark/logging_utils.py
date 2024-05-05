import logging
import logging.config
from logging import Logger
from subprocess import PIPE, STDOUT, Popen
from typing import List, Optional


def setup_logging(
    level: str = "INFO",
    use_colorlog: bool = True,
    format_prefix: Optional[str] = None,
    disable_existing_loggers: bool = False,
    handlers: List[str] = ["console", "file"],
):
    # base logging config
    logging_config = {
        "version": 1,
        "handlers": {
            "console": {"formatter": "simple", "stream": "ext://sys.stdout", "class": "logging.StreamHandler"},
            "file": {"formatter": "simple", "filename": "cli.log", "class": "logging.FileHandler"},
        },
        "root": {"level": level, "handlers": handlers},
        "disable_existing_loggers": disable_existing_loggers,
    }

    # formatters
    logging_config["formatters"] = {
        "simple": {"format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"},
        "colorlog": {
            "()": "colorlog.ColoredFormatter",
            "format": "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            "log_colors": {"DEBUG": "purple", "INFO": "green", "WARNING": "yellow", "CRITICAL": "red", "ERROR": "red"},
        },
    }

    # use colorlog
    if use_colorlog:
        for handler in logging_config["handlers"]:
            logging_config["handlers"][handler]["formatter"] = "colorlog"

    # format prefix
    if format_prefix is not None:
        for formatter in logging_config["formatters"]:
            logging_config["formatters"][formatter]["format"] = (
                f"[{format_prefix}]" + logging_config["formatters"][formatter]["format"]
            )

    logging.config.dictConfig(logging_config)


def run_subprocess_and_log_stream_output(logger: Logger, args):
    popen = Popen(args, stdout=PIPE, stderr=STDOUT)
    for line in iter(popen.stdout.readline, b""):
        if line is not None:
            logger.info(line.decode("utf-8").rstrip())

    popen.wait()
    return popen
