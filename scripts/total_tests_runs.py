import os

import yaml

config_dir = "tests/configs"
config_files = [f for f in os.listdir(config_dir) if not f.startswith("_")]

run_counts = {}
for config_file in config_files:
    with open(os.path.join(config_dir, config_file), "r") as f:
        config = yaml.safe_load(f)

    for default in config.get("defaults", []):
        if isinstance(default, str) and default != "_self_":
            with open(os.path.join(config_dir, f"{default}.yaml"), "r") as f:
                default_config = yaml.safe_load(f)
                params = default_config.get("hydra", {}).get("sweeper", {}).get("params", {})

                if len(params) == 0:
                    run_counts[config_file] = run_counts.get(config_file, 1)
                else:
                    for param_values in params.values():
                        run_counts[config_file] = run_counts.get(config_file, 1) * len(param_values.split(","))


for config_file, run_count in run_counts.items():
    print(f"{config_file}: {run_count} runs")

print(f"Total runs: {sum(run_counts.values())}")
