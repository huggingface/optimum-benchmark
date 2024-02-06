from typing import List, Dict, Any


# TODO: use some kind of logic to handle this instead of this function
def consolidate_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    report = {}

    ## Training

    if "warmup.runtime(s)" in reports[0]:
        report["warmup.runtime(s)"] = reports[0]["warmup.runtime(s)"]
        report["warmup.throughput(samples/s)"] = sum(r["warmup.throughput(samples/s)"] for r in reports)

    if "training.runtime(s)" in reports[0]:
        report["training.runtime(s)"] = reports[0]["training.runtime(s)"]
        report["training.throughput(samples/s)"] = sum(r["training.throughput(samples/s)"] for r in reports)

    if "overall.runtime(s)" in reports[0]:
        report["overall.runtime(s)"] = reports[0]["overall.runtime(s)"]
        report["overall.throughput(samples/s)"] = sum(r["overall.throughput(samples/s)"] for r in reports)

    ## Inference

    if "forward.latency(s)" in reports[0]:
        report["forward.latency(s)"] = reports[0]["forward.latency(s)"]
        report["forward.throughput(samples/s)"] = sum(r["forward.throughput(samples/s)"] for r in reports)

    if "diffusion.throughput(images/s)" in reports[0]:
        report["diffusion.throughput(images/s)"] = sum(r["diffusion.throughput(images/s)"] for r in reports)

    if "forward.peak_memory(MB)" in reports[0]:
        report["forward.max_memory_used(MB)"] = reports[0]["forward.max_memory_used(MB)"]
        report["forward.max_memory_allocated(MB)"] = sum(r["forward.max_memory_allocated(MB)"] for r in reports)
        report["forward.max_memory_reserved(MB)"] = sum(r["forward.max_memory_reserved(MB)"] for r in reports)

    if "forward.energy_consumption(kWh/sample)" in reports[0]:
        report["forward.energy_consumption(kWh/sample)"] = reports[0]["forward.energy_consumption(kWh/sample)"]
        report["forward.carbon_emissions(kgCO2eq/sample)"] = reports[0]["forward.carbon_emissions(kgCO2eq/sample)"]

    if "generate.latency(s)" in reports[0]:
        report["generate.latency(s)"] = reports[0]["generate.latency(s)"]
        report["generate.throughput(tokens/s)"] = sum(r["generate.throughput(tokens/s)"] for r in reports)
        report["decode.latency(s)"] = reports[0]["decode.latency(s)"]
        report["decode.throughput(tokens/s)"] = sum(r["decode.throughput(tokens/s)"] for r in reports)

    if "generate.peak_memory(MB)" in reports[0]:
        report["generate.max_memory_used(MB)"] = reports[0]["generate.max_memory_used(MB)"]
        report["generate.max_memory_allocated(MB)"] = sum(r["generate.max_memory_allocated(MB)"] for r in reports)
        report["generate.max_memory_reserved(MB)"] = sum(r["generate.max_memory_reserved(MB)"] for r in reports)

    if "generate.energy_consumption(kWh/token)" in reports[0]:
        report["generate.energy_consumption(kWh/token)"] = reports[0]["generate.energy_consumption(kWh/token)"]
        report["generate.carbon_emissions(kgCO2eq/token)"] = reports[0]["generate.carbon_emissions(kgCO2eq/token)"]

    return report
