from typing import Any, Dict

from transformers import PreTrainedModel

from ..import_utils import is_peft_available

if is_peft_available():
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING, get_peft_model  # type: ignore


def apply_peft(model: PreTrainedModel, peft_type: str, peft_config: Dict[str, Any]) -> PreTrainedModel:
    peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type](**peft_config)
    return get_peft_model(model=model, peft_config=peft_config)
