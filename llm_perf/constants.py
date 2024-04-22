import pandas as pd

INPUT_SHAPES = {"batch_size": 1, "sequence_length": 256}
GENERATE_KWARGS = {"max_new_tokens": 64, "min_new_tokens": 64}

OPEN_LLM_DATAFRAME = pd.read_csv("hf://datasets/optimum/llm-perf-dataset/open-llm.csv")
PRETRAINED_MODELS_LIST = OPEN_LLM_DATAFRAME.sort_values("Size", ascending=True)["Model"].tolist()

CANONICAL_ORGANIZATIONS = [
    # big companies
    *["google", "facebook", "meta", "meta-llama", "microsoft", "Intel", "TencentARC", "Salesforce"],
    # collectives
    *["EleutherAI", "tiiuae", "NousResearch", "Open-Orca"],
    # HF related
    ["bigcode", "HuggingFaceH4"],
    # community members
    ["teknium"],
    # startups
    *[
        "mistral-community",
        "openai-community",
        "togethercomputer",
        "stabilityai",
        "CohereForAI",
        "databricks",
        "mistralai",
        "internlm",
        "Upstage",
        "xai-org",
        "Phind",
        "01-ai",
        "Deci",
        "Qwen",
    ],
]
CANONICAL_MODELS_LIST = [model for model in PRETRAINED_MODELS_LIST if model.split("/")[0] in CANONICAL_ORGANIZATIONS]
