import os
import traceback
from itertools import product
from logging import getLogger

from llm_perf.utils import (
    CANONICAL_PRETRAINED_OPEN_LLM_LIST,
    GENERATE_KWARGS,
    INPUT_SHAPES,
    OPEN_LLM_LIST,
    PRETRAINED_OPEN_LLM_LIST,
    is_benchmark_conducted,
    is_benchmark_supported,
)
from optimum_benchmark import Benchmark, BenchmarkConfig, BenchmarkReport, InferenceConfig, ProcessConfig, PyTorchConfig
from optimum_benchmark.logging_utils import setup_logging

SUBSET = os.getenv("SUBSET", None)
MACHINE = os.getenv("MACHINE", None)


if os.getenv("MACHINE", None) is None and os.getenv("SUBSET", None) is None:
    PUSH_REPO_ID = "optimum-benchmark/llm-perf-pytorch-cuda-debug"
    CANONICAL_PRETRAINED_OPEN_LLM_LIST = ["gpt2"]
    SUBSET = "unquantized"
elif os.getenv("MACHINE", None) is not None and os.getenv("SUBSET", None) is not None:
    PUSH_REPO_ID = f"optimum-benchmark/llm-perf-pytorch-cuda-{SUBSET}-{MACHINE}"
else:
    raise ValueError("Either both MACHINE and SUBSET should be set for benchmarking or neither for debugging")

ATTENTION_COFIGS = ["eager", "sdpa", "flash_attention_2"]
if SUBSET == "unquantized":
    WEIGHTS_CONFIGS = {
        # unquantized
        "float32": {"torch_dtype": "float32", "quant_scheme": None, "quant_config": {}},
        "float16": {"torch_dtype": "float16", "quant_scheme": None, "quant_config": {}},
        "bfloat16": {"torch_dtype": "bfloat16", "quant_scheme": None, "quant_config": {}},
    }
elif SUBSET == "bnb":
    WEIGHTS_CONFIGS = {
        # bnb
        "4bit-bnb": {"torch_dtype": "float16", "quant_scheme": "bnb", "quant_config": {"load_in_4bit": True}},
        "8bit-bnb": {"torch_dtype": "float16", "quant_scheme": "bnb", "quant_config": {"load_in_8bit": True}},
    }
elif SUBSET == "gptq":
    WEIGHTS_CONFIGS = {
        # gptq
        "4bit-gptq-exllama-v1": {
            "quant_scheme": "gptq",
            "torch_dtype": "float16",
            "quant_config": {"bits": 4, "use_exllama ": True, "version": 1, "model_seqlen": 256},
        },
        "4bit-gptq-exllama-v2": {
            "torch_dtype": "float16",
            "quant_scheme": "gptq",
            "quant_config": {"bits": 4, "use_exllama ": True, "version": 2, "model_seqlen": 256},
        },
    }
elif SUBSET == "awq":
    WEIGHTS_CONFIGS = {
        # awq
        "4bit-awq-gemm": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {"bits": 4, "version": "gemm"},
        },
        "4bit-awq-gemv": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {"bits": 4, "version": "gemv"},
        },
        "4bit-awq-exllama-v1": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {
                "bits": 4,
                "version": "exllama",
                "exllama_config": {"version": 1, "max_input_len": 64, "max_batch_size": 1},
            },
        },
        "4bit-awq-exllama-v2": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {
                "bits": 4,
                "version": "exllama",
                "exllama_config": {"version": 2, "max_input_len": 64, "max_batch_size": 1},
            },
        },
    }


LOGGER = getLogger("llm-perf-backend")
LOGGER.info(f"len(OPEN_LLM_LIST): {len(OPEN_LLM_LIST)}")
LOGGER.info(f"len(PRETRAINED_OPEN_LLM_LIST): {len(PRETRAINED_OPEN_LLM_LIST)}")
LOGGER.info(f"len(CANONICAL_PRETRAINED_OPEN_LLM_LIST): {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)}")


def benchmark_cuda_pytorch(model, attn_implementation, weights_config):
    benchmark_name = f"{weights_config}-{attn_implementation}"
    subfolder = f"{benchmark_name}/{model.replace('/', '--')}"

    torch_dtype = WEIGHTS_CONFIGS[weights_config]["torch_dtype"]
    quant_scheme = WEIGHTS_CONFIGS[weights_config]["quant_scheme"]
    quant_config = WEIGHTS_CONFIGS[weights_config]["quant_config"]

    if not is_benchmark_supported(weights_config, attn_implementation):
        LOGGER.info(f"Skipping benchmark {benchmark_name} with model {model} since it is not supported")
        return

    if is_benchmark_conducted(PUSH_REPO_ID, subfolder):
        LOGGER.info(f"Skipping benchmark {benchmark_name} with model {model} since it was already conducted")
        return

    launcher_config = ProcessConfig(device_isolation=True, device_isolation_action="kill")
    scenario_config = InferenceConfig(
        memory=True,
        energy=True,
        latency=True,
        duration=10,
        iterations=10,
        warmup_runs=10,
        input_shapes=INPUT_SHAPES,
        generate_kwargs=GENERATE_KWARGS,
    )
    backend_config = PyTorchConfig(
        model=model,
        device="cuda",
        device_ids="0",
        no_weights=True,
        library="transformers",
        task="text-generation",
        torch_dtype=torch_dtype,
        quantization_scheme=quant_scheme,
        quantization_config=quant_config,
        attn_implementation=attn_implementation,
        hub_kwargs={"trust_remote_code": True},
    )

    benchmark_config = BenchmarkConfig(
        name=benchmark_name, scenario=scenario_config, launcher=launcher_config, backend=backend_config
    )

    benchmark_config.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)

    try:
        LOGGER.info(f"Running benchmark {benchmark_name} with model {model}")
        benchmark_report = Benchmark.launch(benchmark_config)
        benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
        benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)

    except Exception:
        LOGGER.error(f"Benchmark {benchmark_name} failed with model {model}")
        benchmark_report = BenchmarkReport.from_dict({"traceback": traceback.format_exc()})
        benchmark_report.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)
        benchmark = Benchmark(config=benchmark_config, report=benchmark_report)
        benchmark.push_to_hub(repo_id=PUSH_REPO_ID, subfolder=subfolder, private=True)


if __name__ == "__main__":
    # for isolated process
    os.environ["LOG_TO_FILE"] = "0"
    os.environ["LOG_LEVEL"] = "INFO"

    # for main process
    setup_logging(level="INFO", prefix="MAIN-PROCESS")

    models_attentions_weights = list(
        product(CANONICAL_PRETRAINED_OPEN_LLM_LIST, ATTENTION_COFIGS, WEIGHTS_CONFIGS.keys())
    )

    LOGGER.info(
        f"Running a total of {len(models_attentions_weights)} benchmarks, "
        f"with {len(CANONICAL_PRETRAINED_OPEN_LLM_LIST)} models, "
        f"{len(ATTENTION_COFIGS)} attentions implementations "
        f"and {len(WEIGHTS_CONFIGS)} weights configurations."
    )

    for model, attn_implementation, weights_config in models_attentions_weights:
        benchmark_cuda_pytorch(model, attn_implementation, weights_config)
