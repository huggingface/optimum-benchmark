import logging
import logging.config
import os
from logging import Logger
from subprocess import PIPE, STDOUT, Popen
from typing import Optional

from omegaconf import OmegaConf

API_JOB_LOGGING = {
    "version": 1,
    "formatters": {
        "simple": {"format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"},
        "colorlog": {
            "()": "colorlog.ColoredFormatter",
            "format": "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            "log_colors": {"DEBUG": "purple", "INFO": "green", "WARNING": "yellow", "CRITICAL": "red", "ERROR": "red"},
        },
    },
    "handlers": {"console": {"formatter": "colorlog", "stream": "ext://sys.stdout", "class": "logging.StreamHandler"}},
    "root": {"level": "INFO", "handlers": ["console"]},
    "disable_existing_loggers": False,
}


def setup_logging(level: str = "INFO", prefix: Optional[str] = None):
    if os.environ.get("BENCHMARK_INTERFACE", "API") == "CLI":
        hydra_config = OmegaConf.load(".hydra/hydra.yaml")
        job_logging = OmegaConf.to_container(hydra_config.hydra.job_logging, resolve=True)
    else:
        job_logging = API_JOB_LOGGING.copy()

    job_logging["root"]["level"] = level

    if prefix is not None:
        job_logging["formatters"]["simple"]["format"] = f"[{prefix}]" + job_logging["formatters"]["simple"]["format"]
        job_logging["formatters"]["colorlog"]["format"] = (
            f"[{prefix}]" + job_logging["formatters"]["colorlog"]["format"]
        )

    logging.config.dictConfig(job_logging)


def run_subprocess_and_log_stream_output(logger: Logger, args):
    popen = Popen(args, stdout=PIPE, stderr=STDOUT)
    for line in iter(popen.stdout.readline, b""):
        if line is not None:
            logger.info(line.decode("utf-8").rstrip())

    popen.wait()
    return popen
