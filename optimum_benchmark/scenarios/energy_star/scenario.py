import os
from contextlib import ExitStack

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...backends.base import Backend, BackendConfigT
from ...benchmark.report import BenchmarkReport
from ...import_utils import is_torch_distributed_available
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, Energy, EnergyTracker
from ..base import Scenario
from .config import EnergyStarConfig
from .preprocessing_utils import preprocess

if is_torch_distributed_available():
    import torch.distributed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


PER_TOKEN_BACKENDS = ["pytorch", "onnxruntime", "openvino", "neural-compressor"]

TEXT_GENERATION_DEFAULT_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "temperature": 1.0,
    "do_sample": False,
    "use_cache": True,
    "num_beams": 1,
}
TEXT_GENERATION_PREFILL_OVERRIDES = {
    "max_new_tokens": 1,
    "min_new_tokens": 1,
}
TEXT_GENERATION_WARMUP_OVERRIDES = {
    "max_new_tokens": 2,
    "min_new_tokens": 2,
}

IMAGE_DIFFUSION_DEFAULT_KWARGS = {
    "num_inference_steps": 30,
    "num_images_per_prompt": 1,
}
IMAGE_DIFFUSION_WARMUP_OVERRIDES = {
    "num_inference_steps": 2,
}

TEXT_GENERATION_THROUGHPUT_UNIT = "tokens/s"
IMAGE_DIFFUSION_THROUGHPUT_UNIT = "images/s"
PREPROCESSING_THROUGHPUT_UNIT = "samples/s"
INFERENCE_THROUGHPUT_UNIT = "samples/s"

TEXT_GENERATION_EFFICIENCY_UNIT = "tokens/kWh"
IMAGE_DIFFUSION_EFFICIENCY_UNIT = "images/kWh"
PREPROCESSING_EFFICIENCY_UNIT = "samples/kWh"
INFERENCE_EFFICIENCY_UNIT = "samples/kWh"


class EnergyStarScenario(Scenario[EnergyStarConfig]):
    NAME = "energy_star"

    def __init__(self, config: EnergyStarConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> BenchmarkReport:
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            self.logger.info("\t+ Distributing batch size across processes")
            if self.config.input_shapes["batch_size"] % torch.distributed.get_world_size() != 0:
                raise ValueError(
                    "The batch size must be divisible by the number of processes in a distributed environment"
                )
            self.config.input_shapes["batch_size"] //= torch.distributed.get_world_size()

        self.logger.info("\t+ Loading raw dataset")
        raw_dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.dataset_split,
        )

        if backend.config.task in TEXT_GENERATION_TASKS and backend.pretrained_model.can_generate():
            self.logger.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_DEFAULT_KWARGS, **self.config.generate_kwargs}
            self.logger.info("\t+ Initializing Text Generation report")
            BenchmarkReport.from_list(targets=["preprocess", "load", "prefill", "decode", "per_token"])
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_DEFAULT_KWARGS, **self.config.call_kwargs}
            self.logger.info("\t+ Initializing Image Diffusion report")
            self.report = BenchmarkReport.from_list(targets=["preprocess", "load", "call"])
        else:
            self.logger.info("\t+ Initializing Inference report")
            self.report = BenchmarkReport.from_list(targets=["preprocess", "load", "forward"])

        self.energy_tracker = EnergyTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        self.logger.info("\t+ Preprocessing dataset")
        with self.energy_tracker.track(file_prefix="preprocess"):
            self.dataset = preprocess(
                dataset=raw_dataset,
                task=backend.config.task,
                config=self.config,
                preprocessor=backend.pretrained_processor,
                pretrained_config=backend.pretrained_config,
            )

        self.report.preprocess.energy = self.energy_tracker.get_energy()
        self.report.preprocess.efficiency = Efficiency.from_energy(
            self.report.preprocess.energy,
            volume=self.config.num_samples,
            unit=PREPROCESSING_EFFICIENCY_UNIT,
        )

        self.run_model_loading_tracking(backend)

        self.logger.info("\t+ Initialising dataloader")
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.input_shapes["batch_size"])

        self.logger.info("\t+ Warming up backend for Inference")
        self.sample_inputs = backend.prepare_inputs(next(iter(self.dataloader)))

        for _ in range(self.config.warmup_runs):
            if backend.config.task in TEXT_GENERATION_TASKS and backend.pretrained_model.can_generate():
                _ = backend.generate(
                    self.sample_inputs, {**self.config.generate_kwargs, **TEXT_GENERATION_WARMUP_OVERRIDES}
                )
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                _ = backend.call(self.sample_inputs, {**self.config.call_kwargs, **IMAGE_DIFFUSION_WARMUP_OVERRIDES})
            else:
                _ = backend.forward(self.sample_inputs, self.config.forward_kwargs)

        if backend.config.task in TEXT_GENERATION_TASKS and backend.pretrained_model.can_generate():
            self.logger.info("\t+ Additional warmup for Text Generation")
            _ = backend.generate(
                self.sample_inputs, {**self.config.generate_kwargs, **TEXT_GENERATION_WARMUP_OVERRIDES}
            )
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Additional warmup for Image Diffusion")
            _ = backend.call(self.sample_inputs, {**self.config.call_kwargs, **IMAGE_DIFFUSION_WARMUP_OVERRIDES})

        if self.config.energy:
            if backend.config.task in TEXT_GENERATION_TASKS and backend.pretrained_model.can_generate():
                self.run_text_generation_energy_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_energy_tracking(backend)
            else:
                self.run_inference_energy_tracking(backend)

            self.report.log_energy()
            self.report.log_efficiency()

        return self.report

    # Loading tracking
    def run_model_loading_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running model loading tracking")

        if self.config.energy:
            energy_tracker = EnergyTracker(
                backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
            )

        context_stack = ExitStack()
        if self.config.energy:
            context_stack.enter_context(energy_tracker.track())

        with context_stack:
            self.logger.info("\t+ Loading model for Inference")
            backend.load()

        if self.config.energy:
            self.report.load.energy = energy_tracker.get_energy()

    ## Energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info(f"\t+ Running Text Generation energy tracking for {self.config.iterations} iterations")

        prefill_measures = []

        prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}

        for k in range(self.config.iterations):
            self.logger.info(f"\t+ Prefill iteration {k+1}/{self.config.iterations}")
            with self.energy_tracker.track(file_prefix="prefill"):
                prefill_volume = 0
                for inputs in tqdm(self.dataloader):
                    inputs = backend.prepare_inputs(inputs)
                    _ = backend.prefill(inputs, prefill_kwargs)
                    try:
                        prefill_volume += inputs["input_ids"].size(dim=1) * self.config.input_shapes["batch_size"]
                    except KeyError:
                        prefill_volume += 1
            prefill_measures.append(self.energy_tracker.get_energy())

        prefill_energy = Energy.aggregate(prefill_measures)
        self.report.prefill.energy = prefill_energy
        self.report.prefill.efficiency = Efficiency.from_energy(
            prefill_energy, prefill_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )
        self.report.prefill.measures = prefill_measures

        generate_measures = []
        for k in range(self.config.iterations):
            self.logger.info(f"\t+ Decoding iteration {k+1}/{self.config.iterations}")
            with self.energy_tracker.track(file_prefix="generate"):
                for inputs in tqdm(self.dataloader):
                    inputs = backend.prepare_inputs(inputs)
                    _ = backend.generate(inputs, self.config.generate_kwargs)
            generate_measures.append(self.energy_tracker.get_energy())

        generate_energy = Energy.aggregate(generate_measures)
        decode_energy = generate_energy - prefill_energy
        decode_volume = self.atomic_decode_volume

        self.report.decode.energy = decode_energy
        self.report.decode.efficiency = Efficiency.from_energy(
            decode_energy, decode_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )
        self.report.decode.measures = [
            generate_measures[i] - prefill_measures[i] for i in range(self.config.iterations)
        ]

    def run_image_diffusion_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info(f"\t+ Running Image Diffusion energy tracking for {self.config.iterations} iterations")

        measures = []

        for k in range(self.config.iterations):
            self.logger.info(f"\t+ Iteration {k+1}/{self.config.iterations}")
            with self.energy_tracker.track(file_prefix="call"):
                for inputs in tqdm(self.dataloader):
                    inputs = backend.prepare_inputs(inputs)
                    _ = backend.call(inputs, self.config.call_kwargs)
            measures.append(self.energy_tracker.get_energy())

        call_energy = Energy.aggregate(measures)
        call_volume = self.atomic_call_volume

        self.report.call.energy = call_energy
        self.report.call.efficiency = Efficiency.from_energy(
            call_energy, call_volume, unit=IMAGE_DIFFUSION_EFFICIENCY_UNIT
        )
        self.report.call.measures = measures

    def run_inference_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info(f"\t+ Running Inference energy tracking for {self.config.iterations} iterations")

        measures = []

        for k in range(self.config.iterations):
            self.logger.info(f"\t+ Iteration {k+1}/{self.config.iterations}")
            with self.energy_tracker.track(file_prefix="forward"):
                for inputs in tqdm(self.dataloader):
                    inputs = backend.prepare_inputs(inputs)
                    _ = backend.forward(inputs, self.config.forward_kwargs)
            measures.append(self.energy_tracker.get_energy())

        forward_energy = Energy.aggregate(measures)
        forward_volume = self.atomic_forward_volume

        self.report.forward.energy = forward_energy
        self.report.forward.efficiency = Efficiency.from_energy(
            forward_energy, forward_volume, unit=INFERENCE_EFFICIENCY_UNIT
        )
        self.report.forward.measures = measures

    @property
    def atomic_forward_volume(self) -> int:  # in samples
        return self.config.num_samples

    @property
    def atomic_call_volume(self) -> int:  # in images
        if "prompt" in self.sample_inputs:
            return self.config.num_samples * self.config.call_kwargs["num_images_per_prompt"]
        else:
            return self.config.num_samples

    @property
    def atomic_decode_volume(self) -> int:  # in tokens
        return (
            self.config.num_samples
            * self.config.generate_kwargs["num_beams"]  # at each beam stage there are num_beams tokens generated
            * (self.config.generate_kwargs["max_new_tokens"] - 1)  # 1 token is generated during prefill
        )
