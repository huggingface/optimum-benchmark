import os
from itertools import product
from logging import getLogger

import pandas as pd

from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.benchmarks.report import BenchmarkReport
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.logging_utils import setup_logging

CWD = os.getcwd()
MACHINE = os.getenv("MACHINE", "1xA100")
SUBSET = os.getenv("SUBSET", "unquantized")
PUSH_REPO_ID = f"optimum-benchmark/llm-perf-pytorch-cuda-{SUBSET}-{MACHINE}"

OPEN_LLM_DF = pd.read_csv("hf://datasets/optimum/llm-perf-dataset/open-llm.csv")
MODELS_LIST = OPEN_LLM_DF.sort_values("Size", ascending=True)["Model"].tolist()

GENERATE_KWARGS = {"max_new_tokens": 64, "min_new_tokens": 64}
INPUT_SHAPES = {"batch_size": 1, "sequence_length": 256}

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
                "exllama_config": {"version": 1, "max_input_len": 256, "max_batch_size": 1},
            },
        },
        "4bit-awq-exllama-v2": {
            "torch_dtype": "float16",
            "quant_scheme": "awq",
            "quant_config": {
                "bits": 4,
                "version": "exllama",
                "exllama_config": {"version": 2, "max_input_len": 256, "max_batch_size": 1},
            },
        },
    }


setup_logging()
LOGGER = getLogger("llm-perf-backend")


def benchmark_cuda_pytorch():
    print(f"Total number of experiments: {len(list(product(MODELS_LIST, ATTENTION_COFIGS, WEIGHTS_CONFIGS.keys())))}")

    launcher_config = ProcessConfig(start_method="spawn", device_isolation=True, device_isolation_action="kill")
    benchmark_config = InferenceConfig(
        memory=True,
        energy=True,
        latency=True,
        duration=10,
        iterations=10,
        warmup_runs=10,
        input_shapes=INPUT_SHAPES,
        generate_kwargs=GENERATE_KWARGS,
    )

    for model, attn_implementation, weights_config in product(MODELS_LIST, ATTENTION_COFIGS, WEIGHTS_CONFIGS.keys()):
        torch_dtype = WEIGHTS_CONFIGS[weights_config]["torch_dtype"]
        quant_scheme = WEIGHTS_CONFIGS[weights_config]["quant_scheme"]
        quant_config = WEIGHTS_CONFIGS[weights_config]["quant_config"]

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
        )

        experiment_name = f"{weights_config}-{attn_implementation}"
        subfolder = f"{experiment_name}/{model.replace('/', '--')}"

        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            benchmark=benchmark_config,
            launcher=launcher_config,
            backend=backend_config,
        )

        try:
            # skip if the same experiment is already uploaded, and its benchmark report is already uploaded
            loaded_experiment_config = ExperimentConfig.from_pretrained(repo_id=PUSH_REPO_ID, subfolder=subfolder)
            if loaded_experiment_config.to_dict() == experiment_config.to_dict():
                BenchmarkReport.from_pretrained(repo_id=PUSH_REPO_ID, subfolder=subfolder)
                LOGGER.info(
                    f"Skipping {experiment_name} with model {model} since the same "
                    "experiment is already uploaded with its benchmark report"
                )
                continue
        except Exception:
            pass

        experiment_config.push_to_hub(subfolder=subfolder, repo_id=PUSH_REPO_ID, private=True)

        try:
            benchmark_report = launch(experiment_config)
            benchmark_report.push_to_hub(subfolder=subfolder, repo_id=PUSH_REPO_ID, private=True)
        except Exception as e:
            os.chdir(CWD)  # go back to original_dir
            LOGGER.error(f"Experiment {experiment_name} failed with model {model}")

            if "torch.cuda.OutOfMemoryError" in str(e):
                benchmark_report = BenchmarkReport.from_targets(["decode", "prefill", "per_token", "error"])
                benchmark_report.error = "CUDA: Out of memory"
                benchmark_report.push_to_hub(subfolder=subfolder, repo_id=PUSH_REPO_ID, private=True)
            elif "gptq" in str(e) and "assert outfeatures % 32 == 0" in str(e):
                benchmark_report = BenchmarkReport.from_targets(["decode", "prefill", "per_token", "error"])
                benchmark_report.error = "GPTQ: assert outfeatures % 32 == 0"
                benchmark_report.push_to_hub(subfolder=subfolder, repo_id=PUSH_REPO_ID, private=True)
            else:
                LOGGER.error(f"Unknown error: {e}")

            continue


if __name__ == "__main__":
    benchmark_cuda_pytorch()
