import os
import logging
import logging.config
from subprocess import Popen, PIPE, STDOUT

from omegaconf import OmegaConf

JOB_LOGGING = {
    "version": 1,
    "formatters": {
        "simple": {"format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"},
        "colorlog": {
            "()": "colorlog.ColoredFormatter",
            "format": "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            "log_colors": {
                "DEBUG": "purple",
                "INFO": "green",
                "WARNING": "yellow",
                "CRITICAL": "red",
                "ERROR": "red",
            },
        },
    },
    "handlers": {
        "console": {
            "formatter": "colorlog",
            "stream": "ext://sys.stdout",
            "class": "logging.StreamHandler",
        },
        "file": {
            "filename": "api.log",
            "formatter": "simple",
            "class": "logging.FileHandler",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
    "disable_existing_loggers": False,
}


def setup_logging(level: str = "INFO"):
    if os.path.exists(".hydra/hydra.yaml"):
        hydra_config = OmegaConf.load(".hydra/hydra.yaml")
        job_logging = OmegaConf.to_container(
            hydra_config.hydra.job_logging,
            resolve=True,
        )
    else:
        job_logging = JOB_LOGGING.copy()

    job_logging["root"]["level"] = level
    logging.config.dictConfig(job_logging)


def run_process_and_log_stream_output(logger, args):
    popen = Popen(args, stdout=PIPE, stderr=STDOUT)
    for line in iter(popen.stdout.readline, b""):
        if line is not None:
            logger.info(line.decode("utf-8").rstrip())

    popen.wait()
    return popen
