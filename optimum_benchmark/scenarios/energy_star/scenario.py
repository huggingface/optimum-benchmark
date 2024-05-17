from dataclasses import dataclass
from logging import getLogger

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...backends.base import Backend, BackendConfigT
from ...benchmark.report import BenchmarkMeasurements, BenchmarkReport
from ...import_utils import is_torch_distributed_available
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, EnergyTracker
from ..base import Scenario
from .config import EnergyStarConfig
from .preprocessing_utils import preprocess

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("energy_star")


PER_TOKEN_BACKENDS = ["pytorch", "onnxruntime", "openvino", "neural-compressor"]

TEXT_GENERATION_DEFAULT_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "temperature": 1.0,
    "do_sample": False,
    "use_cache": True,
    "pad_token_id": 0,
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
INFERENCE_THROUGHPUT_UNIT = "samples/s"

TEXT_GENERATION_EFFICIENCY_UNIT = "tokens/kWh"
IMAGE_DIFFUSION_EFFICIENCY_UNIT = "images/kWh"
INFERENCE_EFFICIENCY_UNIT = "samples/kWh"


@dataclass
class TextGenerationReport(BenchmarkReport):
    preprocess: BenchmarkMeasurements
    per_token: BenchmarkMeasurements
    prefill: BenchmarkMeasurements
    decode: BenchmarkMeasurements


@dataclass
class ImageDiffusionReport(BenchmarkReport):
    preprocess: BenchmarkMeasurements
    call: BenchmarkMeasurements


@dataclass
class InferenceReport(BenchmarkReport):
    preprocess: BenchmarkMeasurements
    forward: BenchmarkMeasurements


class EnergyStarScenario(Scenario[EnergyStarConfig]):
    NAME = "energy_star"

    def __init__(self, config: EnergyStarConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> BenchmarkReport:
        if is_torch_distributed_available() and torch.distributed.is_initialized():
            LOGGER.info("\t+ Distributing batch size across processes")
            if self.config.input_shapes["batch_size"] % torch.distributed.get_world_size() != 0:
                raise ValueError(
                    "The batch size must be divisible by the number of processes in a distributed environment"
                )
            self.config.input_shapes["batch_size"] //= torch.distributed.get_world_size()

        self.energy_tracker = EnergyTracker(device=backend.config.device, device_ids=backend.config.device_ids)

        LOGGER.info("\t+ Loading dataset")
        raw_dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.dataset_split,
        )

        LOGGER.info("\t+ Preprocessing dataset")
        with self.energy_tracker.track():
            self.dataset = preprocess(
                dataset=raw_dataset,
                task=backend.config.task,
                config=self.config,
                preprocessor=backend.pretrained_processor,
            )
        self.preprocessing_energy = self.energy_tracker.get_energy()
        self.energy_tracker.reset()

        LOGGER.info("\t+ Initialising dataloader")
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.input_shapes["batch_size"])

        if backend.config.task in TEXT_GENERATION_TASKS:
            LOGGER.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_WARMUP_OVERRIDES, **self.config.generate_kwargs}
            LOGGER.info("\t+ Initializing Text Generation report")
            self.report = TextGenerationReport(
                preprocess=BenchmarkMeasurements(),
                per_token=BenchmarkMeasurements(),
                prefill=BenchmarkMeasurements(),
                decode=BenchmarkMeasurements(),
            )
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_WARMUP_OVERRIDES, **self.config.call_kwargs}
            LOGGER.info("\t+ Initializing Image Diffusion report")
            self.report = ImageDiffusionReport(preprocess=BenchmarkMeasurements(), call=BenchmarkMeasurements())

        else:
            LOGGER.info("\t+ Initializing Inference report")
            self.report = InferenceReport(preprocess=BenchmarkMeasurements(), forward=BenchmarkMeasurements())

        self.report.preprocess.energy = self.preprocessing_energy
        self.report.preprocess.efficiency = Efficiency.from_energy(
            self.report.preprocess.energy, self.inference_volume, unit=INFERENCE_EFFICIENCY_UNIT
        )

        LOGGER.info("\t+ Preparing backend for Inference")
        backend.prepare_for_inference(
            **backend.model_shapes,
            **self.config.input_shapes,
            **self.config.generate_kwargs,
            **self.config.forward_kwargs,
            **self.config.call_kwargs,
        )

        LOGGER.info("\t+ Warming up backend for Inference")
        warmup_inputs = backend.prepare_inputs(next(iter(self.dataloader)))
        for _ in range(self.config.warmup_runs):
            if backend.config.task in TEXT_GENERATION_TASKS:
                _ = backend.generate(warmup_inputs, {"max_new_tokens": 2, "min_new_tokens": 2})
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                _ = backend.call(warmup_inputs, {"num_inference_steps": 2})
            else:
                _ = backend.forward(warmup_inputs, self.config.forward_kwargs)

        if backend.config.task in TEXT_GENERATION_TASKS:
            LOGGER.info("\t+ Additional warmup for Text Generation")
            _ = backend.generate(warmup_inputs, self.config.generate_kwargs)
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Additional warmup for Image Diffusion")
            _ = backend.call(warmup_inputs, self.config.call_kwargs)

        if self.config.energy:
            if backend.config.task in TEXT_GENERATION_TASKS:
                self.run_text_generation_energy_tracking(backend)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                self.run_image_diffusion_energy_tracking(backend)
            else:
                self.run_inference_energy_tracking(backend)

            self.report.log_energy()
            self.report.log_efficiency()

        return self.report

    ## Energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")

        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.forward(inputs, self.config.forward_kwargs)

        self.report.prefill.energy = self.energy_tracker.get_energy()
        self.report.prefill.efficiency = Efficiency.from_energy(
            self.report.prefill.energy, self.text_generation_prefill_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )
        self.energy_tracker.reset()

        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                _ = backend.generate(inputs, self.config.generate_kwargs)

        self.report.decode.energy = self.energy_tracker.get_energy()
        self.report.decode.efficiency = Efficiency.from_energy(
            self.report.decode.energy, self.text_generation_decode_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )
        self.energy_tracker.reset()

    def run_image_diffusion_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")

        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.call(self.call_inputs, self.config.call_kwargs)

        self.report.call.energy = self.energy_tracker.get_energy()
        self.report.call.efficiency = Efficiency.from_energy(
            self.report.call.energy, self.image_diffusion_volume, unit=IMAGE_DIFFUSION_EFFICIENCY_UNIT
        )
        self.energy_tracker.reset()

    def run_inference_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")

        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.forward(inputs, self.config.forward_kwargs)

        self.report.forward.energy = self.energy_tracker.get_energy()
        self.report.forward.efficiency = Efficiency.from_energy(
            self.report.forward.energy, self.inference_volume, unit=INFERENCE_EFFICIENCY_UNIT
        )
        self.energy_tracker.reset()

    @property
    def inference_volume(self) -> int:  # in samples
        return self.config.num_samples

    @property
    def image_diffusion_volume(self) -> int:  # in images
        return self.config.input_shapes["batch_size"] * self.config.call_kwargs["num_images_per_prompt"]

    @property
    def text_generation_prefill_volume(self) -> int:  # in tokens
        return self.config.input_shapes["batch_size"] * self.config.input_shapes["sequence_length"]

    @property
    def text_generation_per_token_volume(self) -> int:  # in tokens
        return self.config.input_shapes["batch_size"] * self.config.generate_kwargs["num_return_sequences"]

    @property
    def text_generation_decode_volume(self) -> int:  # in tokens
        return (
            self.config.input_shapes["batch_size"]
            * self.config.generate_kwargs["num_return_sequences"]
            * (self.config.generate_kwargs["max_new_tokens"] - 1)  # 1 token is generated during prefill
        )
