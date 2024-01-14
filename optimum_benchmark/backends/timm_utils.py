from transformers import PretrainedConfig

from ..import_utils import is_timm_available

if is_timm_available():
    import timm


def get_pretrained_config(model_name: str) -> PretrainedConfig:
    model_source, model_name = timm.models.parse_model_name(model_name)
    if model_source == "hf-hub":
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = timm.models.load_model_config_from_hf(model_name)
    else:
        pretrained_cfg = timm.get_pretrained_cfg(model_name)

    return pretrained_cfg
