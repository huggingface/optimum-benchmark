from dataclasses import dataclass
from logging import getLogger

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...backends.base import Backend, BackendConfigT
from ...import_utils import is_torch_distributed_available
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, EnergyTracker
from ..base import Benchmark
from ..report import BenchmarkMeasurements, BenchmarkReport
from ..utils import compute_call_volume, compute_decode_volume, compute_forward_volume, compute_prefill_volume
from .config import EnergyStarConfig
from .preprocessing_utils import preprocess

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("energy_star")

# let's define energy star's specific kwargs and units instead of using inference benchmark's

PER_TOKEN_BACKENDS = ["pytorch", "onnxruntime", "openvino", "neural-compressor"]

IMAGE_DIFFUSION_KWARGS = {"num_inference_steps": 30, "num_images_per_prompt": 1}
TEXT_GENERATION_KWARGS = {
    "num_return_sequences": 1,
    "max_new_tokens": 100,
    "min_new_tokens": 100,
    "temperature": 1.0,
    "do_sample": False,
    "use_cache": True,
    "num_beams": 1,
    # "pad_token_id": 0, # not needed for energy star
}


TEXT_GENERATION_THROUGHPUT_UNIT = "tokens/s"
IMAGE_DIFFUSION_THROUGHPUT_UNIT = "images/s"
INFERENCE_THROUGHPUT_UNIT = "samples/s"

TEXT_GENERATION_EFFICIENCY_UNIT = "tokens/kWh"
IMAGE_DIFFUSION_EFFICIENCY_UNIT = "images/kWh"
INFERENCE_EFFICIENCY_UNIT = "samples/kWh"


@dataclass
class TextGenerationReport(BenchmarkReport):
    prefill: BenchmarkMeasurements
    decode: BenchmarkMeasurements
    per_token: BenchmarkMeasurements
    preprocess: BenchmarkMeasurements


@dataclass
class ImageDiffusionReport(BenchmarkReport):
    call: BenchmarkMeasurements
    preprocess: BenchmarkMeasurements


@dataclass
class InferenceReport(BenchmarkReport):
    forward: BenchmarkMeasurements
    preprocess: BenchmarkMeasurements


class EnergyStarBenchmark(Benchmark[EnergyStarConfig]):
    NAME = "energy_star"

    def __init__(self, config: EnergyStarConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT][BackendConfigT]) -> None:
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
            self.config.generate_kwargs = {**TEXT_GENERATION_KWARGS, **self.config.generate_kwargs}
            LOGGER.info("\t+ Initializing Text Generation report")
            self.report = TextGenerationReport(
                decode=BenchmarkMeasurements(),
                prefill=BenchmarkMeasurements(),
                per_token=BenchmarkMeasurements(),
                preprocess=BenchmarkMeasurements(),
            )

        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_KWARGS, **self.config.call_kwargs}
            LOGGER.info("\t+ Initializing Image Diffusion report")
            self.report = ImageDiffusionReport(call=BenchmarkMeasurements(), preprocess=BenchmarkMeasurements())

        else:
            LOGGER.info("\t+ Initializing Inference report")
            self.report = InferenceReport(forward=BenchmarkMeasurements(), preprocess=BenchmarkMeasurements())

        self.report.preprocess.energy = self.preprocessing_energy
        self.report.preprocess.efficiency = Efficiency.from_energy(
            self.report.preprocess.energy, self.config.num_samples, unit=INFERENCE_EFFICIENCY_UNIT
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
                warmup_kwargs = self.config.generate_kwargs.copy()
                warmup_kwargs.update({"max_new_tokens": 1, "min_new_tokens": 1})
                _ = backend.generate(warmup_inputs, warmup_kwargs)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                warmup_kwargs = self.config.call_kwargs.copy()
                warmup_kwargs.update({"num_inference_steps": 2})
                _ = backend.call(warmup_inputs, warmup_kwargs)
            else:
                warmup_kwargs = self.config.forward_kwargs.copy()
                _ = backend.forward(warmup_inputs, warmup_kwargs)

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

    ## Energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")

        prefill_kwargs = self.config.generate_kwargs.copy()
        prefill_kwargs.update({"max_new_tokens": 1, "min_new_tokens": 1})

        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.prefill(inputs, prefill_kwargs)

        self.report.prefill.energy = self.energy_tracker.get_energy()

        prefill_tokens_volume = compute_prefill_volume(
            backend.config.task, self.config.input_shapes, self.config.generate_kwargs
        )
        self.report.prefill.efficiency = Efficiency.from_energy(
            self.report.prefill.energy,
            volume=prefill_tokens_volume * self.config.num_samples,
            unit=TEXT_GENERATION_EFFICIENCY_UNIT,
        )

        self.energy_tracker.reset()
        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                _ = backend.generate(inputs, self.config.generate_kwargs)

        self.report.decode.energy = self.energy_tracker.get_energy()

        decode_tokens_volume = compute_decode_volume(self.config.input_shapes, self.config.generate_kwargs)
        self.report.decode.efficiency = Efficiency.from_energy(
            self.report.decode.energy,
            volume=decode_tokens_volume * self.config.num_samples,
            unit=TEXT_GENERATION_EFFICIENCY_UNIT,
        )
        self.energy_tracker.reset()

    def run_image_diffusion_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")

        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.call(inputs, self.config.call_kwargs)

        self.report.call.energy = self.energy_tracker.get_energy()

        image_diffusion_volume = compute_call_volume(self.config.input_shapes, self.config.call_kwargs)
        self.report.call.efficiency = Efficiency.from_energy(
            self.report.call.energy,
            volume=image_diffusion_volume * self.config.num_samples,
            unit=IMAGE_DIFFUSION_EFFICIENCY_UNIT,
        )
        self.energy_tracker.reset()

    def run_inference_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running energy tracking")

        with self.energy_tracker.track():
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.forward(inputs, self.config.forward_kwargs)

        self.report.forward.energy = self.energy_tracker.get_energy()

        inference_samples_volume = compute_forward_volume(self.config.input_shapes)
        self.report.forward.efficiency = Efficiency.from_energy(
            self.report.forward.energy,
            volume=inference_samples_volume * self.config.num_samples,
            unit=INFERENCE_EFFICIENCY_UNIT,
        )

        self.energy_tracker.reset()

    def get_report(self) -> InferenceReport:
        return self.report
