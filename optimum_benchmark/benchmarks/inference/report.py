from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List
from logging import getLogger

from ..report import BenchmarkReport

LOGGER = getLogger("report")


@dataclass
class InferenceReport(BenchmarkReport):
    # Config
    batch_size: int
    # Metrics
    forward: Dict[str, Any] = field(default_factory=dict)

    # POPULATING
    def populate_latency(self, forward_latencies_list: List[float]):
        ## Latency
        self.forward["latency"] = {
            "list[s]": forward_latencies_list,
            "mean(s)": compute_mean(forward_latencies_list),
            "stdev(s)": compute_stdev(forward_latencies_list),
        }
        ## Throughput
        forward_throughputs_list = [self.batch_size / latency for latency in forward_latencies_list]
        self.forward["throughput"] = {
            "list[samples/s]": forward_throughputs_list,
            "mean(samples/s)": compute_mean(forward_throughputs_list),
            "stdev(samples/s)": compute_stdev(forward_throughputs_list),
        }

    def populate_memory(self, forward_memories_dict: Dict[str, Any]):
        self.forward["memory"] = forward_memories_dict

    def populate_energy(self, forward_energies_dict: Dict[str, Any]):
        self.forward["energy"] = forward_energies_dict

    # LOGGING
    def log_latency(self):
        for key, value in self.forward["latency"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ forward.latency.{key}: {value:f} (s)")
        for key, value in self.forward["throughput"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ forward.throughput.{key}: {value:f} (samples/s)")

    def log_memory(self):
        for key, value in self.forward["memory"].items():
            LOGGER.info(f"\t+ forward.memory.{key}: {value:f} (MB)")

    def log_energy(self):
        for key, value in self.forward["energy"].items():
            LOGGER.info(f"\t+ forward.energy.{key}: {value:f} (kWh)")

    def log_all(self) -> None:
        if "latency" in self.forward:
            self.log_latency()
        if "memory" in self.forward:
            self.log_memory()
        if "energy" in self.forward:
            self.log_energy()

    # add operator to aggregate multiple reports
    def __add__(self, other: "InferenceReport") -> "InferenceReport":
        agg_report = InferenceReport(batch_size=self.batch_size + other.batch_size)
        if "latency" in self.forward and "latency" in other.forward:
            agg_forward_latencies_list = [
                (lat_1 + lat_2) / 2
                for lat_1, lat_2 in zip(self.forward["latency"]["list[s]"], other.forward["latency"]["list[s]"])
            ]
            agg_report.populate_latency(agg_forward_latencies_list)

        if "memory" in self.forward and "memory" in other.forward:
            agg_forward_memories_dict = {}
            for key in self.forward["memory"]:
                if "vram" in key:
                    # our vram measures are not process-specific
                    agg_forward_memories_dict[key] = max(self.forward["memory"][key], other.forward["memory"][key])
                else:
                    # ram and pytorch measures are process-specific
                    agg_forward_memories_dict[key] = self.forward["memory"][key] + other.forward["memory"][key]

            agg_report.populate_memory(agg_forward_memories_dict)

        if "energy" in self.forward and "energy" in other.forward:
            agg_forward_energies_dict = {}
            for key in self.forward["energy"]:
                # theoretically, the energies measured by codecarbon are process-specific (it's not clear from the code)
                agg_forward_energies_dict[key] = self.forward["energy"][key] + other.forward["energy"][key]

            agg_report.populate_energy(agg_forward_energies_dict)

        return agg_report


@dataclass
class ImageDiffusionReport(BenchmarkReport):
    # Config
    batch_size: int
    num_images_per_prompts: int
    # Metrics
    call: Dict[str, Any] = field(default_factory=dict)

    # POPULATING
    def populate_latency(self, call_latencies_list: List[float]):
        ## Latency
        self.call["latency"] = {
            "list[s]": call_latencies_list,
            "mean(s)": compute_mean(call_latencies_list),
            "stdev(s)": compute_stdev(call_latencies_list),
        }
        ## Throughput
        call_throughputs_list = [
            self.batch_size * self.num_images_per_prompts / latency for latency in call_latencies_list
        ]
        self.call["throughput"] = {
            "list[images/s]": call_throughputs_list,
            "mean[images/s]": compute_mean(call_throughputs_list),
            "stdev[images/s]": compute_stdev(call_throughputs_list),
        }

    def populate_memory(self, call_memories_dict: Dict[str, Any]):
        self.call["memory"] = call_memories_dict

    def populate_energy(self, call_energies_dict: Dict[str, Any]):
        self.call["energy"] = call_energies_dict

    # LOGGING
    def log_latency(self):
        for key, value in self.call["latency"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ call.latency.{key}: {value:f} (s)")
        for key, value in self.call["throughput"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ call.throughput.{key}: {value:f} (images/s)")

    def log_memory(self):
        for key, value in self.call["memory"].items():
            LOGGER.info(f"\t+ call.memory.{key}: {value:f} (MB)")

    def log_energy(self):
        for key, value in self.call["energy"].items():
            LOGGER.info(f"\t+ call.energy.{key}: {value:f} (kWh)")

    def log_all(self) -> None:
        if "latency" in self.call:
            self.log_latency()
        if "memory" in self.call:
            self.log_memory()
        if "energy" in self.call:
            self.log_energy()

    # add operator to aggregate multiple reports
    def __add__(self, other: "ImageDiffusionReport") -> "ImageDiffusionReport":
        assert self.num_images_per_prompts == other.num_images_per_prompts, "num_images_per_prompts must be the same"

        agg_report = ImageDiffusionReport(
            batch_size=self.batch_size + other.batch_size,
            num_images_per_prompts=self.num_images_per_prompts,
        )
        if "latency" in self.call and "latency" in other.call:
            agg_call_latencies_list = [
                (lat_1 + lat_2) / 2
                for lat_1, lat_2 in zip(self.call["latency"]["list[s]"], other.call["latency"]["list[s]"])
            ]
            agg_report.populate_latency(agg_call_latencies_list)

        if "memory" in self.call and "memory" in other.call:
            agg_call_memories_dict = {}
            for key in self.call["memory"]:
                if "vram" in key:
                    # our vram measures are not process-specific
                    agg_call_memories_dict[key] = max(self.call["memory"][key], other.call["memory"][key])
                else:
                    # ram and pytorch measures are process-specific
                    agg_call_memories_dict[key] = self.call["memory"][key] + other.call["memory"][key]

            agg_report.populate_memory(agg_call_memories_dict)

        if "energy" in self.call and "energy" in other.call:
            agg_call_energies_dict = {}
            for key in self.call["energy"]:
                # theoretically, the energies measured by codecarbon are process-specific (it's not clear from the code)
                agg_call_energies_dict[key] = self.call["energy"][key] + other.call["energy"][key]

            agg_report.populate_energy(agg_call_energies_dict)

        return agg_report


@dataclass
class TextGenerationReport(BenchmarkReport):
    # Config
    batch_size: int
    sequence_length: int
    num_new_tokens: int
    num_return_sequences: int
    # Prefill Metrics
    prefill: Dict[str, Any] = field(default_factory=dict)
    # Decode Metrics
    decode: Dict[str, Any] = field(default_factory=dict)

    def populate_latency(self, forward_latencies_list: List[float], generate_latencies_list: List[float]):
        ## Latency
        self.prefill["latency"] = {
            "list[s]": forward_latencies_list,
            "mean(s)": compute_mean(forward_latencies_list),
            "stdev(s)": compute_stdev(forward_latencies_list),
        }
        ## Throughput
        prefill_throughputs_list = [
            self.batch_size * self.sequence_length / latency for latency in forward_latencies_list
        ]
        self.prefill["throughput"] = {
            "list[tokens/s]": prefill_throughputs_list,
            "mean[tokens/s]": compute_mean(prefill_throughputs_list),
            "stdev[tokens/s]": compute_stdev(prefill_throughputs_list),
        }
        ## Latency
        decode_latencies_list = [
            generate_latency - self.prefill["latency"]["mean(s)"] for generate_latency in generate_latencies_list
        ]
        self.decode["latency"] = {
            "list[s]": decode_latencies_list,
            "mean(s)": compute_mean(decode_latencies_list),
            "stdev(s)": compute_stdev(decode_latencies_list),
        }
        ## Throughput
        decode_throughputs_list = [
            self.batch_size * self.num_new_tokens * self.num_return_sequences / latency
            for latency in decode_latencies_list
        ]
        self.decode["throughput"] = {
            "list[tokens/s]": decode_throughputs_list,
            "mean[tokens/s]": compute_mean(decode_throughputs_list),
            "stdev[tokens/s]": compute_stdev(decode_throughputs_list),
        }

    def populate_memory(self, forward_memories_dict: Dict[str, Any], generate_memories_dict: Dict[str, Any]):
        self.prefill["memory"] = forward_memories_dict
        self.decode["memory"] = generate_memories_dict

    def populate_energy(self, forward_energies_dict: Dict[str, Any], generate_energies_dict: Dict[str, Any]):
        self.prefill["energy"] = forward_energies_dict
        self.decode["energy"] = generate_energies_dict

    # LOGGING
    def log_latency(self):
        for key, value in self.prefill["latency"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ prefill.latency.{key}: {value:f} (s)")
        for key, value in self.prefill["throughput"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ prefill.throughput.{key}: {value:f} (tokens/s)")
        for key, value in self.decode["latency"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ decode.latency.{key}: {value:f} (s)")
        for key, value in self.decode["throughput"].items():
            if "list" in key:
                continue
            LOGGER.info(f"\t+ decode.throughput.{key}: {value:f} (tokens/s)")

    def log_memory(self):
        for key, value in self.prefill["memory"].items():
            LOGGER.info(f"\t+ prefill.memory.{key}: {value:f} (MB)")
        for key, value in self.decode["memory"].items():
            LOGGER.info(f"\t+ decode.memory.{key}: {value:f} (MB)")

    def log_energy(self):
        for key, value in self.prefill["energy"].items():
            LOGGER.info(f"\t+ prefill.energy.{key}: {value:f} (kWh)")
        for key, value in self.decode["energy"].items():
            LOGGER.info(f"\t+ decode.energy.{key}: {value:f} (kWh)")

    def log_all(self) -> None:
        if "latency" in self.prefill:
            self.log_latency()
        if "memory" in self.prefill:
            self.log_memory()
        if "energy" in self.prefill:
            self.log_energy()

    # add operator to aggregate multiple reports
    def __add__(self, other: "TextGenerationReport") -> "TextGenerationReport":
        agg_report = TextGenerationReport(
            batch_size=self.batch_size + other.batch_size,
            sequence_length=self.sequence_length,
            num_new_tokens=self.num_new_tokens,
            num_return_sequences=self.num_return_sequences,
        )
        if "latency" in self.prefill and "latency" in other.prefill:
            agg_forward_latencies_list = [
                (lat_1 + lat_2) / 2
                for lat_1, lat_2 in zip(self.prefill["latency"]["list[s]"], other.prefill["latency"]["list[s]"])
            ]
            agg_generate_latencies_list = [
                (lat_1 + lat_2) / 2
                for lat_1, lat_2 in zip(self.decode["latency"]["list[s]"], other.decode["latency"]["list[s]"])
            ]
            agg_report.populate_latency(agg_forward_latencies_list, agg_generate_latencies_list)

        if "memory" in self.prefill and "memory" in other.prefill:
            agg_forward_memories_dict = {}
            for key in self.prefill["memory"]:
                if "vram" in key:
                    # our vram measures are not process-specific
                    agg_forward_memories_dict[key] = max(self.prefill["memory"][key], other.prefill["memory"][key])
                else:
                    # ram and pytorch measures are process-specific
                    agg_forward_memories_dict[key] = self.prefill["memory"][key] + other.prefill["memory"][key]

            agg_generate_memories_dict = {}
            for key in self.decode["memory"]:
                if "vram" in key:
                    # our vram measures are not process-specific
                    agg_generate_memories_dict[key] = max(self.decode["memory"][key], other.decode["memory"][key])
                else:
                    # ram and pytorch measures are process-specific
                    agg_generate_memories_dict[key] = self.decode["memory"][key] + other.decode["memory"][key]

            agg_report.populate_memory(agg_forward_memories_dict, agg_generate_memories_dict)

        if "energy" in self.prefill and "energy" in other.prefill:
            agg_forward_energies_dict = {}
            for key in self.prefill["energy"]:
                # theoretically, the energies measured by codecarbon are process-specific (it's not clear from the code)
                agg_forward_energies_dict[key] = self.prefill["energy"][key] + other.prefill["energy"][key]

            agg_generate_energies_dict = {}
            for key in self.decode["energy"]:
                # theoretically, the energies measured by codecarbon are process-specific (it's not clear from the code)
                agg_generate_energies_dict[key] = self.decode["energy"][key] + other.decode["energy"][key]

            agg_report.populate_energy(agg_forward_energies_dict, agg_generate_energies_dict)

        return agg_report


def compute_mean(values: List[float]) -> float:
    return mean(values) if len(values) > 0 else 0.0


def compute_stdev(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0
