from typing import Type

from ..import_utils import is_peft_available

if is_peft_available():
    from peft import (
        IA3Config,
        LoraConfig,
        PeftConfig,
        AdaLoraConfig,
        PrefixTuningConfig,
        PromptEncoderConfig,
        PromptLearningConfig,
    )


PEFT_TASKS_TYPES = [
    "SEQ_CLS",
    "SEQ_2_SEQ_LM",
    "CAUSAL_LM",
    "TOKEN_CLS",
    "QUESTION_ANS",
    "FEATURE_EXTRACTION",
]

PEFT_CONFIG = {
    "base_model_name_or_path": None,
    "revision": None,  # str
    "peft_type": None,  # PeftType: can't be changed anyway
    "task_type": None,  # TaskType: SEQ_CLS, SEQ_2_SEQ_LM, CAUSAL_LM, TOKEN_CLS, QUESTION_ANS, FEATURE_EXTRACTION
    "inference_mode": False,
}
LORA_CONFIG = {
    **PEFT_CONFIG,  # inherits from PEFT_CONFIG
    "auto_mapping": None,  # dict
    "r": 8,  # int
    "target_modules": None,  # List[str] | str
    "lora_alpha": 8,  # int
    "lora_dropout": 0,  # float
    "fan_in_fan_out": False,  # bool
    "bias": "none",  # str
    "modules_to_save": None,  # List[str]
    "init_lora_weights": True,  # bool
    "layers_to_transform": None,  # List[int] | int
    "layers_pattern": None,  # str
}
ADA_LORA_CONFIG = {
    **LORA_CONFIG,  # inherits from LORA_CONFIG
    "target_r": None,  # int
    "init_r": None,  # int
    "tinit": None,  # int
    "tfinal": None,  # int
    "deltaT": None,  # int
    "beta1": None,  # float
    "beta2": None,  # float
    "orth_reg_weight": None,  # float
    "total_step": None,  # Optional[int]
    "rank_pattern": None,  # Optional[dict]
}
PROMPT_TUNING_CONFIG = {
    **PEFT_CONFIG,  # inherits from PEFT_CONFIG
    "num_virtual_tokens": None,  # int
    "token_dim": None,  # int
    "num_transformer_submodules": None,  # int
    "num_attention_heads": None,  # int
    "num_layers": None,  # int
}
PREFIX_TUNING_CONFIG = {
    **PROMPT_TUNING_CONFIG,  # inherits from PROMPT_TUNING_CONFIG
    "encoder_hidden_size": None,  # int
    "prefix_projection": False,  # bool
}
P_TUNING_CONFIG = {
    **PROMPT_TUNING_CONFIG,  # inherits from PROMPT_TUNING_CONFIG
    "encoder_reparameterization_type": None,  # Union[str, PromptEncoderReparameterizationType]
    "encoder_hidden_size": None,  # int
    "encoder_num_layers": None,  # int
    "encoder_dropout": None,  # float
}
IA3_CONFIG = {
    **PEFT_CONFIG,  # inherits from PEFT_CONFIG
    "target_modules": None,  # List[str] | str
    "feedforward_modules": None,  # List[str] | str
    "fan_in_fan_out": False,  # bool
    "modules_to_save": None,  # List[str]
    "init_ia3_weights": True,  # bool
}
PEFT_CONFIGS = {
    "lora": LORA_CONFIG,
    "prefix_tuning": PREFIX_TUNING_CONFIG,
    "prompt_tuning": PROMPT_TUNING_CONFIG,
    "p_tuning": P_TUNING_CONFIG,
    "ada_lora": ADA_LORA_CONFIG,
    "ia3": IA3_CONFIG,
}


def get_peft_config_class(peft_strategy: str) -> Type["PeftConfig"]:
    if peft_strategy == "lora":
        return LoraConfig
    elif peft_strategy == "ada_lora":
        return AdaLoraConfig
    elif peft_strategy == "prompt_tuning":
        return PromptLearningConfig
    elif peft_strategy == "prefix_tuning":
        return PrefixTuningConfig
    elif peft_strategy == "p_tuning":
        return PromptEncoderConfig
    elif peft_strategy == "ia3":
        return IA3Config
