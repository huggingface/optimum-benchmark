from optimum_benchmark.benchmarks.report import BenchmarkReport


def common_errors_reporter(error, logger, subfolder, push_repo_id):
    benchmark_report = BenchmarkReport.from_targets(["decode", "prefill", "per_token", "error"])

    if "torch.cuda.OutOfMemoryError" in str(error):
        logger.error("CUDA: Out of memory")
        benchmark_report.error = "CUDA: Out of memory"
        benchmark_report.push_to_hub(subfolder=subfolder, repo_id=push_repo_id, private=True)
    elif "gptq" in str(error) and "assert outfeatures % 32 == 0" in str(error):
        logger.error("GPTQ: assert outfeatures % 32 == 0")
        benchmark_report.error = "GPTQ: assert outfeatures % 32 == 0"
        benchmark_report.push_to_hub(subfolder=subfolder, repo_id=push_repo_id, private=True)
    elif "gptq" in str(error) and "assert infeatures % self.group_size == 0" in str(error):
        logger.error("GPTQ: assert infeatures % self.group_size == 0")
        benchmark_report.error = "GPTQ: assert infeatures % self.group_size == 0"
        benchmark_report.push_to_hub(subfolder=subfolder, repo_id=push_repo_id, private=True)
    elif "support Flash Attention 2.0" in str(error):
        logger.error("Flash Attention 2.0: not supported yet")
        benchmark_report.error = "Flash Attention 2.0: not supported yet"
        benchmark_report.push_to_hub(subfolder=subfolder, repo_id=push_repo_id, private=True)
    elif "support an attention implementation through torch.nn.functional.scaled_dot_product_attention" in str(error):
        logger.error("SDPA: not supported yet")
        benchmark_report.error = "SDPA: not supported yet"
        benchmark_report.push_to_hub(subfolder=subfolder, repo_id=push_repo_id, private=True)
    elif "FlashAttention only support fp16 and bf16 data type" in str(error):
        logger.error("FlashAttention: only support fp16 and bf16 data type")
        benchmark_report.error = "FlashAttention: only support fp16 and bf16 data type"
        benchmark_report.push_to_hub(subfolder=subfolder, repo_id=push_repo_id, private=True)
    else:
        logger.error(f"Unknown error: {error}")


def is_experiment_conducted(experiment_config, push_repo_id, subfolder):
    try:
        loaded_experiment_config = experiment_config.from_pretrained(repo_id=push_repo_id, subfolder=subfolder)
        if loaded_experiment_config.to_dict() == experiment_config.to_dict():
            BenchmarkReport.from_pretrained(repo_id=push_repo_id, subfolder=subfolder)
            return True
    except Exception:
        pass

    return False


def is_experiment_not_supported(torch_dtype, attn_implementation):
    if attn_implementation == "flash_attention_2" and torch_dtype == "float32":
        return True

    return False
