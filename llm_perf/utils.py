from typing import Tuple

import pandas as pd

from optimum_benchmark.report import BenchmarkReport

OPEN_LLM_LEADERBOARD = pd.read_csv("hf://datasets/optimum-benchmark/open-llm-leaderboard/open-llm-leaderboard.csv")


INPUT_SHAPES = {"batch_size": 1, "sequence_length": 256}
GENERATE_KWARGS = {"max_new_tokens": 64, "min_new_tokens": 64}


OPEN_LLM_LIST = OPEN_LLM_LEADERBOARD.drop_duplicates(subset=["Model"])["Model"].tolist()
PRETRAINED_OPEN_LLM_LIST = (
    OPEN_LLM_LEADERBOARD[OPEN_LLM_LEADERBOARD["Type"] == "pretrained"]
    .drop_duplicates(subset=["Model"])["Model"]
    .tolist()
)
# CANONICAL_ORGANIZATIONS = [
#     # big companies
#     *["google", "facebook", "meta", "meta-llama", "microsoft", "Intel", "TencentARC", "Salesforce"],
#     # collectives
#     *["EleutherAI", "tiiuae", "NousResearch", "Open-Orca"],
#     # HF related
#     ["bigcode", "HuggingFaceH4", "huggyllama"],
#     # community members
#     ["teknium"],
#     # startups
#     *[
#         "mistral-community",
#         "openai-community",
#         "togethercomputer",
#         "stabilityai",
#         "CohereForAI",
#         "databricks",
#         "mistralai",
#         "internlm",
#         "Upstage",
#         "xai-org",
#         "Phind",
#         "01-ai",
#         "Deci",
#         "Qwen",
#     ],
# ]
# CANONICAL_PRETRAINED_OPEN_LLM_LIST = [model for model in PRETRAINED_OPEN_LLM_LIST if model.split("/")[0] in CANONICAL_ORGANIZATIONS]
CANONICAL_PRETRAINED_OPEN_LLM_LIST = [
    "01-ai/Yi-34B",
    "01-ai/Yi-6B",
    "Deci/DeciLM-7B",
    "Deci/DeciCoder-1b",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
    "EleutherAI/polyglot-ko-12.8b",
    "EleutherAI/pythia-1.3b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-2.7b",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-6.7b",
    "EleutherAI/pythia-70m",
    "Qwen/Qwen-7B",
    "Qwen/Qwen-14B",
    "Qwen/Qwen-72B",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-14B",
    "Qwen/Qwen1.5-32B",
    "Qwen/Qwen1.5-72B",
    "Qwen/Qwen1.5-110B",
    "Qwen/Qwen1.5-MoE-A2.7B",
    "Qwen/Qwen2-beta-14B",
    "Qwen/Qwen2-beta-72B",
    "Salesforce/codegen-6B-nl",
    "Salesforce/codegen-16B-nl",
    "TencentARC/Mistral_Pro_8B_v0.1",
    "databricks/dbrx-base",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",
    "facebook/xglm-564M",
    "facebook/xglm-4.5B",
    "facebook/xglm-7.5B",
    "google/gemma-2b",
    "google/gemma-7b",
    "google/recurrentgemma-2b",
    "google/recurrentgemma-7b",
    "internlm/internlm-20b",
    "internlm/internlm2-20b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "microsoft/phi-1_5",
    "microsoft/rho-math-1b-v0.1",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "openai-community/gpt2",
    "openai-community/gpt2-large",
    "stabilityai/stablelm-2-12b",
    "stabilityai/stablelm-2-1_6b",
    "stabilityai/stablelm-3b-4e1t",
    "stabilityai/stablelm-base-alpha-3b",
    "stabilityai/stablelm-base-alpha-7b",
    "tiiuae/falcon-rw-1b",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "tiiuae/falcon-180B",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1",
]


def errors_handler(error: str) -> Tuple[bool, BenchmarkReport]:
    valid_error = True
    benchmark_report = BenchmarkReport.from_list(["error"])

    if "torch.cuda.OutOfMemoryError" in error:
        benchmark_report.error = "CUDA: Out of memory"
    elif "gptq" in error and "assert outfeatures % 32 == 0" in error:
        benchmark_report.error = "GPTQ: assert outfeatures % 32 == 0"
    elif "gptq" in error and "assert infeatures % self.group_size == 0" in error:
        benchmark_report.error = "GPTQ: assert infeatures % self.group_size == 0"
    elif "support Flash Attention 2.0" in error:
        benchmark_report.error = "Flash Attention 2.0: not supported yet"
    elif "support an attention implementation through torch.nn.functional.scaled_dot_product_attention" in error:
        benchmark_report.error = "SDPA: not supported yet"
    elif "FlashAttention only support fp16 and bf16 data type" in error:
        benchmark_report.error = "FlashAttention: only support fp16 and bf16 data type"
    else:
        benchmark_report.error = f"Unknown error: {error}"
        valid_error = False

    return valid_error, benchmark_report


def is_benchmark_conducted(benchmark_config, push_repo_id, subfolder):
    try:
        loaded_benchmark_config = benchmark_config.from_pretrained(repo_id=push_repo_id, subfolder=subfolder)
        loaded_benchmark_dict = loaded_benchmark_config.to_dict()
        benchmark_dict = benchmark_config.to_dict()
        loaded_benchmark_dict.pop("environment")
        benchmark_dict.pop("environment")
        if loaded_benchmark_dict == benchmark_dict:
            BenchmarkReport.from_pretrained(repo_id=push_repo_id, subfolder=subfolder)
            return True
    except Exception:
        pass

    return False


def is_benchmark_supported(weights_config, attn_implementation):
    if attn_implementation == "flash_attention_2" and weights_config == "float32":
        return False

    return True
