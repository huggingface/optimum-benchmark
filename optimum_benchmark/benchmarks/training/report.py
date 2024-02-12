from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List
from logging import getLogger

from ..report import BenchmarkReport

LOGGER = getLogger("report")


@dataclass
class TrainingReport(BenchmarkReport):
    max_steps: int
    warmup_steps: int
    num_processes: int
    per_process_batch_size: int
    gradient_accumulation_steps: int

    training: Dict[str, Any] = field(default_factory=dict)

    # POPULATING
    def populate_latency(self, all_latencies_list: List[float]) -> None:
        assert (
            len(all_latencies_list) == self.max_steps
        ), f"Expected {self.max_steps} latencies, but got {len(all_latencies_list)} latencies"
        ## Latency
        training_latencies_list = all_latencies_list[self.warmup_steps :]
        self.training["latency"] = {
            "list[s/step]": training_latencies_list,
            "mean(s/step)": compute_mean(training_latencies_list),
            "stdev(s/step)": compute_stdev(training_latencies_list),
        }
        ## Throughput
        training_throughputs_list = [
            self.per_process_batch_size * self.gradient_accumulation_steps / latency
            for latency in training_latencies_list
        ]
        self.training["throughput"] = {
            "list[samples/s]": training_throughputs_list,
            "mean(samples/s)": compute_mean(training_throughputs_list),
            "stdev(samples/s)": compute_stdev(training_throughputs_list),
        }

    def populate_memory(self, all_memories_dict: Dict[str, float]) -> None:
        ## Memory
        self.training["memory"] = all_memories_dict

    def populate_energy(self, all_energies_dict: Dict[str, float]) -> None:
        ## Energy
        self.training["energy"] = all_energies_dict

    # LOGGING
    def log_latency(self):
        for key, value in self.training["latency"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ training.latency.{key}: {value:f} (s)")
        for key, value in self.training["throughput"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ training.throughput.{key}: {value:f} (samples/s)")

    def log_memory(self):
        for key, value in self.training["memory"].items():
            LOGGER.info(f"\t+ training.memory.{key}: {value:f} (MB)")

    def log_energy(self):
        for key, value in self.training["energy"].items():
            LOGGER.info(f"\t+ training.energy.{key}: {value:f} (kWh)")

    def log_all(self):
        if "latency" in self.training:
            self.log_latency()
        if "memory" in self.training:
            self.log_memory()
        if "energy" in self.training:
            self.log_energy()

    def __add__(self, other: "TrainingReport") -> "TrainingReport":
        assert self.max_steps == other.max_steps, "Both reports must have the same max_steps"
        assert self.warmup_steps == other.warmup_steps, "Both reports must have the same warmup_steps"
        assert (
            self.gradient_accumulation_steps == other.gradient_accumulation_steps
        ), "Both reports must have the same gradient_accumulation_steps"

        agg_report = TrainingReport(
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            num_processes=self.num_processes + other.num_processes,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_process_batch_size=self.per_process_batch_size + other.per_process_batch_size,
        )

        if "latency" in self.training and "latency" in other.training:
            agg_training_latencies_list = [
                max(lat_1, lat_2)
                for lat_1, lat_2 in zip(self.training["latency"]["list[s]"], other.training["latency"]["list[s]"])
            ]
            agg_report.populate_latency(agg_training_latencies_list)

        if "memory" in self.training and "memory" in other.training:
            agg_training_memories_dict = {}
            for key in self.training["memory"]:
                if "vram" in key:
                    # our vram measures are not process-specific
                    agg_training_memories_dict[key] = max(self.training["memory"][key], other.training["memory"][key])
                else:
                    # ram and pytorch measures are process-specific
                    agg_training_memories_dict[key] = self.training["memory"][key] + other.training["memory"][key]

            agg_report.populate_memory(agg_training_memories_dict)

        if "energy" in self.training and "energy" in other.training:
            agg_training_energies_dict = {}
            for key in self.training["energy"]:
                # theoretically, the energies measured by codecarbon are process-specific (it's not clear from the code)
                agg_training_energies_dict[key] = self.training["energy"][key] + other.training["energy"][key]

            agg_report.populate_energy(agg_training_energies_dict)

        return agg_report


def compute_mean(values: List[float]) -> float:
    return mean(values) if len(values) > 0 else 0.0


def compute_stdev(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0
