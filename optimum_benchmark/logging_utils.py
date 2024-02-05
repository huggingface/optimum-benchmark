import logging.config
import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


def get_colorlog_dict():
    plugins_spec = importlib.util.find_spec("hydra_plugins.hydra_colorlog")
    plugins_path = Path(plugins_spec.origin).parent / "conf/hydra/"
    colorlog_conf = OmegaConf.load(plugins_path / "hydra_logging/colorlog.yaml")
    colorlog_dict = OmegaConf.to_container(colorlog_conf, resolve=True)
    return colorlog_dict


def setup_colorlog_logging():
    colorlog_dict = get_colorlog_dict()
    logging.config.dictConfig(colorlog_dict)
