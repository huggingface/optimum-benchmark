import os
import logging.config

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


def setup_colorlog_logging() -> None:
    if os.path.exists(".hydra/hydra.yaml"):
        hydra_config = OmegaConf.load(".hydra/hydra.yaml")
        job_logging = OmegaConf.to_container(
            hydra_config.hydra.job_logging,
            resolve=True,
        )
        logging.config.dictConfig(job_logging)
    else:
        logging.config.dictConfig(JOB_LOGGING)
