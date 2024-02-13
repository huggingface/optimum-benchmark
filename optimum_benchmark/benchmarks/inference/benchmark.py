from logging import getLogger
from typing import List, Tuple, Dict

from ..base import Benchmark
from .config import InferenceConfig
from ...trackers.energy import EnergyTracker
from ...trackers.memory import MemoryTracker
from ...trackers.latency import LatencyTracker
from ...backends.base import Backend, BackendConfigT
from ...generators.input_generator import InputGenerator
from ...import_utils import is_torch_distributed_available
from ...task_utils import TEXT_GENERATION_TASKS, IMAGE_DIFFUSION_TASKS
from .report import InferenceReport, TextGenerationReport, ImageDiffusionReport

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("inference")

IMAGE_DIFFUSION_KWARGS = {
    "num_inference_steps": 30,
    "num_images_per_prompt": 1,
}

TEXT_GENERATION_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "temperature": 1.0,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": 0,
    "num_beams": 1,
}


class InferenceBenchmark(Benchmark[InferenceConfig]):
    NAME = "inference"

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> None:
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            if self.config.input_shapes["batch_size"] % torch.distributed.get_world_size() != 0:
                raise ValueError(
                    "The batch size must be divisible by the number of processes in a distributed environment"
                )
            self.config.input_shapes["batch_size"] //= torch.distributed.get_world_size()

        LOGGER.info("\t+ Creating input generator")
        self.input_generator = InputGenerator(
            task=backend.config.task,
            model_shapes=backend.model_shapes,
            input_shapes=self.config.input_shapes,
        )

        if backend.config.task in TEXT_GENERATION_TASKS:
            LOGGER.info("\t+ Generating and preparing Text Generation input")
            self.forward_inputs = self.input_generator(mode="forward")
            self.generate_input = self.input_generator(mode="generate")
            self.forward_inputs = backend.prepare_inputs(self.forward_inputs)
            self.generate_input = backend.prepare_inputs(self.generate_input)
            LOGGER.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_KWARGS, **self.config.generate_kwargs}
            LOGGER.info("\t+ Initializing Text Generation report")
            self.report = TextGenerationReport(
                batch_size=self.config.input_shapes["batch_size"],
                sequence_length=self.config.input_shapes["sequence_length"],
                num_new_tokens=self.config.generate_kwargs["max_new_tokens"],
                num_return_sequences=self.config.generate_kwargs["num_return_sequences"],
            )

        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Generating and preparing Image Diffusion input")
            self.diffuse_input = self.input_generator(mode="call")
            self.diffuse_input = backend.prepare_inputs(self.diffuse_input)
            LOGGER.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.forward_kwargs = {**IMAGE_DIFFUSION_KWARGS, **self.config.forward_kwargs}
            LOGGER.info("\t+ Initializing Image Diffusion report")
            self.report = ImageDiffusionReport(
                batch_size=self.config.input_shapes["batch_size"],
                num_images_per_prompts=self.config.forward_kwargs["num_images_per_prompt"],
            )

        else:
            LOGGER.info("\t+ Generating and preparing Inference input")
            self.forward_inputs = self.input_generator(mode="forward")
            self.forward_inputs = backend.prepare_inputs(self.forward_inputs)
            LOGGER.info("\t+ Initializing Inference report")
            self.report = InferenceReport(
                batch_size=self.config.input_shapes["batch_size"],
            )

        LOGGER.info("\t+ Preparing backend for Inference")
        backend.prepare_for_inference(
            **backend.model_shapes,
            **self.config.input_shapes,
            **self.config.forward_kwargs,
            **self.config.generate_kwargs,
        )

        LOGGER.info("\t+ Warming up backend for Inference")
        for _ in range(self.config.warmup_runs):
            if backend.config.task in TEXT_GENERATION_TASKS:
                generate_warmup_kwargs = {"max_new_tokens": 2, "min_new_tokens": 2}
                _ = backend.generate(self.generate_input, generate_warmup_kwargs)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                diffuse_warmup_kwargs = {"num_inference_steps": 2}
                _ = backend.call(self.diffuse_input, diffuse_warmup_kwargs)
            else:
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        if self.config.memory:
            LOGGER.info("\t+ Creating inference memory tracker")
            self.memory_tracker = MemoryTracker(
                backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
            )
            if backend.config.task in TEXT_GENERATION_TASKS:
                forward_memories_dict, generate_memories_dict = self.run_text_generation_memory_tracking(backend)
                self.report.populate_memory(forward_memories_dict, generate_memories_dict)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                call_memories_dict = self.run_image_diffusion_memory_tracking(backend)
                self.report.populate_memory(call_memories_dict)
            else:
                forward_memories_dict = self.run_inference_memory_tracking(backend)
                self.report.populate_memory(forward_memories_dict)

            self.report.log_memory()

        if self.config.latency:
            LOGGER.info("\t+ Creating inference latency tracker")
            self.latency_tracker = LatencyTracker(backend=backend.config.name, device=backend.config.device)
            if backend.config.task in TEXT_GENERATION_TASKS:
                forward_latencies_dict, generate_latencies_dict = self.run_text_generation_latency_tracking(backend)
                self.report.populate_latency(forward_latencies_dict, generate_latencies_dict)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                call_latencies_dict = self.run_image_diffusion_latency_tracking(backend)
                self.report.populate_latency(call_latencies_dict)
            else:
                forward_latencies_dict = self.run_latency_inference_tracking(backend)
                self.report.populate_latency(forward_latencies_dict)

            self.report.log_latency()

        if self.config.energy:
            LOGGER.info("\t+ Creating inference energy tracker")
            self.energy_tracker = EnergyTracker(device=backend.config.device, device_ids=backend.config.device_ids)
            if backend.config.task in TEXT_GENERATION_TASKS:
                forward_energies_dict, generate_energies_dict = self.run_text_generation_energy_tracking(backend)
                self.report.populate_energy(forward_energies_dict, generate_energies_dict)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                call_energies_dict = self.run_image_diffusion_energy_tracking(backend)
                self.report.populate_energy(call_energies_dict)
            else:
                forward_energies_dict = self.run_inference_energy_tracking(backend)
                self.report.populate_energy(forward_energies_dict)

            self.report.log_energy()

    ## Memory tracking
    def run_text_generation_memory_tracking(self, backend: Backend) -> Tuple[Dict[str, float], Dict[str, float]]:
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        forward_memories_dict = self.memory_tracker.get_memories_dict()

        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.generate(self.generate_input, self.config.generate_kwargs)

        generate_memories_dict = self.memory_tracker.get_memories_dict()

        return forward_memories_dict, generate_memories_dict

    def run_image_diffusion_memory_tracking(self, backend: Backend) -> Dict[str, float]:
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.call(self.diffuse_input, self.config.forward_kwargs)

        call_memories_dict = self.memory_tracker.get_memories_dict()

        return call_memories_dict

    def run_inference_memory_tracking(self, backend: Backend) -> Dict[str, float]:
        LOGGER.info("\t+ Running memory tracking")
        self.memory_tracker.reset()
        with self.memory_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        forward_memories_dict = self.memory_tracker.get_memories_dict()

        return forward_memories_dict

    ## Latency tracking
    def run_text_generation_latency_tracking(self, backend: Backend) -> Tuple[List[float], List[float]]:
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker.reset()
        while self.latency_tracker.get_total_latency() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        forward_latencies_list = self.latency_tracker.get_latencies_list()

        self.latency_tracker.reset()
        while self.latency_tracker.get_total_latency() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.generate(self.generate_input, self.config.generate_kwargs)

        generate_latencies_list = self.latency_tracker.get_latencies_list()

        return forward_latencies_list, generate_latencies_list

    def run_image_diffusion_latency_tracking(self, backend: Backend) -> List[float]:
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker.reset()
        while self.latency_tracker.get_total_latency() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.call(self.diffuse_input, self.config.forward_kwargs)

        call_latencies_list = self.latency_tracker.get_latencies_list()

        return call_latencies_list

    def run_latency_inference_tracking(self, backend: Backend) -> List[float]:
        LOGGER.info("\t+ Running latency tracking")
        self.latency_tracker.reset()
        while self.latency_tracker.get_total_latency() < self.config.duration:
            with self.latency_tracker.track():
                _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        forward_latencies_list = self.latency_tracker.get_latencies_list()

        return forward_latencies_list

    ## Energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend) -> Tuple[Dict[str, float], Dict[str, float]]:
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        forward_energies_dict = self.energy_tracker.get_energies_dict()

        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.generate(self.generate_input, self.config.generate_kwargs)

        generate_energies_dict = self.energy_tracker.get_energies_dict()

        return forward_energies_dict, generate_energies_dict

    def run_image_diffusion_energy_tracking(self, backend: Backend) -> Dict[str, float]:
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.call(self.diffuse_input, self.config.forward_kwargs)

        call_energies_dict = self.energy_tracker.get_energies_dict()

        return call_energies_dict

    def run_inference_energy_tracking(self, backend: Backend) -> Dict[str, float]:
        LOGGER.info("\t+ Running energy tracking")
        self.energy_tracker.reset()
        with self.energy_tracker.track():
            _ = backend.forward(self.forward_inputs, self.config.forward_kwargs)

        forward_energies_dict = self.energy_tracker.get_energies_dict()

        return forward_energies_dict

    def get_report(self) -> InferenceReport:
        return self.report
