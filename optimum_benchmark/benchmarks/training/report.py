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
    per_process_batch_size: int
    gradient_accumulation_steps: int

    overall: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    warmup: Dict[str, Any] = field(default_factory=dict)

    world_size: int = 1

    # POPULATING
    def populate_latency(self, overall_latencies_list: List[float]) -> None:
        assert (
            len(overall_latencies_list) == self.max_steps
        ), f"Expected {self.max_steps} latencies, but got {len(overall_latencies_list)} latencies"
        # Overall
        ## Latency
        self.overall["latency"] = {
            "list[s/step]": overall_latencies_list,
            "mean(s/step)": compute_mean(overall_latencies_list),
            "stdev(s/step)": compute_stdev(overall_latencies_list),
        }
        ## Throughput
        overall_throughputs_list = [
            self.world_size * self.per_process_batch_size * self.gradient_accumulation_steps / latency
            for latency in overall_latencies_list
        ]
        self.overall["throughput"] = {
            "list[samples/s]": overall_throughputs_list,
            "mean(samples/s)": compute_mean(overall_throughputs_list),
            "stdev(samples/s)": compute_stdev(overall_throughputs_list),
        }
        # Training
        ## Latency
        training_latencies_list = overall_latencies_list[self.warmup_steps :]
        self.training["latency"] = {
            "list[s/step]": training_latencies_list,
            "mean(s/step)": compute_mean(training_latencies_list),
            "stdev(s/step)": compute_stdev(training_latencies_list),
        }
        ## Throughput
        training_throughputs_list = overall_throughputs_list[self.warmup_steps :]
        self.training["throughput"] = {
            "list[samples/s]": training_throughputs_list,
            "mean(samples/s)": compute_mean(training_throughputs_list),
            "stdev(samples/s)": compute_stdev(training_throughputs_list),
        }
        # Warmup
        ## Latency
        warmup_latencies_list = overall_latencies_list[: self.warmup_steps]
        self.warmup["latency"] = {
            "list[s/step]": warmup_latencies_list,
            "mean(s/step)": compute_mean(warmup_latencies_list),
            "stdev(s/step)": compute_stdev(warmup_latencies_list),
        }
        ## Throughput
        warmup_throughputs_list = overall_throughputs_list[: self.warmup_steps]
        self.warmup["throughput"] = {
            "list[samples/s]": warmup_throughputs_list,
            "mean(samples/s)": compute_mean(warmup_throughputs_list),
            "stdev(samples/s)": compute_stdev(warmup_throughputs_list),
        }

    def populate_memory(self, overall_memories_dict: Dict[str, float]) -> None:
        self.warmup["memory"] = overall_memories_dict
        self.overall["memory"] = overall_memories_dict
        self.training["memory"] = overall_memories_dict

    def populate_energy(self, overall_energies_dict: Dict[str, float]) -> None:
        self.overall["energy"] = overall_energies_dict
        # can't get training only or warmup only energies
        # self.warmup["energy"] = overall_energies_dict
        # self.training["energy"] = overall_energies_dict
        # TODO: use a callback for energy instead of a tracker

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
        for key, value in self.overall["energy"].items():
            LOGGER.info(f"\t+ overall.energy.{key}: {value:f} (kWh)")

    def log_all(self):
        if "latency" in self.training:
            self.log_latency()
        if "memory" in self.training:
            self.log_memory()
        if "energy" in self.training:
            self.log_energy()

    # LOGIC
    def __add__(self, other: "TrainingReport") -> "TrainingReport":
        assert self.max_steps == other.max_steps, "Both reports must have the same max_steps"
        assert self.warmup_steps == other.warmup_steps, "Both reports must have the same warmup_steps"
        assert (
            self.gradient_accumulation_steps == other.gradient_accumulation_steps
        ), "Both reports must have the same gradient_accumulation_steps"

        agg_report = TrainingReport(
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            world_size=self.world_size + other.world_size,
            per_process_batch_size=self.per_process_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        if "latency" in self.overall:
            agg_overall_latencies_list = [
                max(lat_1, lat_2)
                for lat_1, lat_2 in zip(
                    self.overall["latency"]["list[s/step]"], other.overall["latency"]["list[s/step]"]
                )
            ]
            agg_report.populate_latency(agg_overall_latencies_list)

        if "memory" in self.overall:
            agg_overall_memories_dict = {}
            for key in self.overall["memory"]:
                if "vram" in key:
                    # our vram measures are not process-specific
                    agg_overall_memories_dict[key] = max(self.overall["memory"][key], other.overall["memory"][key])
                else:
                    # ram and pytorch measures are process-specific (can be accumulated)
                    agg_overall_memories_dict[key] = self.overall["memory"][key] + other.overall["memory"][key]

            agg_report.populate_memory(agg_overall_memories_dict)

        if "energy" in self.overall:
            agg_overall_energies_dict = {}
            for key in self.overall["energy"]:
                # theoretically, the energies measured by codecarbon are process-specific (it's not clear from the code)
                agg_overall_energies_dict[key] = self.overall["energy"][key] + other.overall["energy"][key]

            agg_report.populate_energy(agg_overall_energies_dict)

        return agg_report


def compute_mean(values: List[float]) -> float:
    return mean(values) if len(values) > 0 else 0.0


def compute_stdev(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0
