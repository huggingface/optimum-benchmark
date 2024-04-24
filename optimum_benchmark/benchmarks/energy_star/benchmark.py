import os
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
from .config import EnergyStarConfig
from .preprocessing_utils import preprocess

if is_torch_distributed_available():
    import torch.distributed

LOGGER = getLogger("energy_star")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
PREPROCESSING_THROUGHPUT_UNIT = "samples/s"
INFERENCE_THROUGHPUT_UNIT = "samples/s"

TEXT_GENERATION_EFFICIENCY_UNIT = "tokens/kWh"
IMAGE_DIFFUSION_EFFICIENCY_UNIT = "images/kWh"
PREPROCESSING_EFFICIENCY_UNIT = "samples/kWh"
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

        LOGGER.info("\t+ Loading raw dataset")
        raw_dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            split=self.config.dataset_split,
        )

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

        self.energy_tracker = EnergyTracker(
            backend=backend.config.name, device=backend.config.device, device_ids=backend.config.device_ids
        )

        LOGGER.info("\t+ Preprocessing dataset")
        with self.energy_tracker.track(file_prefix="preprocess"):
            self.dataset = preprocess(
                dataset=raw_dataset,
                task=backend.config.task,
                config=self.config,
                preprocessor=backend.pretrained_processor,
                pretrained_config= backend.pretrained_config,
            )

        self.report.preprocess.energy = self.energy_tracker.get_energy()
        self.report.preprocess.efficiency = Efficiency.from_energy(
            self.report.preprocess.energy,
            volume=self.config.num_samples,
            unit=PREPROCESSING_EFFICIENCY_UNIT,
        )

        LOGGER.info("\t+ Preparing backend for Inference")
        backend.prepare_for_inference(
            **backend.model_shapes,
            **self.config.input_shapes,
            **self.config.generate_kwargs,
            **self.config.forward_kwargs,
            **self.config.call_kwargs,
        )

        LOGGER.info("\t+ Initialising dataloader")
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.input_shapes["batch_size"])

        LOGGER.info("\t+ Warming up backend for Inference")
        self.sample_inputs = backend.prepare_inputs(next(iter(self.dataloader)))

        for _ in range(self.config.warmup_runs):
            if backend.config.task in TEXT_GENERATION_TASKS:
                warmup_kwargs = self.config.generate_kwargs.copy()
                warmup_kwargs.update({"max_new_tokens": 2, "min_new_tokens": 2})
                _ = backend.generate(self.sample_inputs, warmup_kwargs)
            elif backend.config.task in IMAGE_DIFFUSION_TASKS:
                warmup_kwargs = self.config.call_kwargs.copy()
                warmup_kwargs.update({"num_inference_steps": 2})
                _ = backend.call(self.sample_inputs, warmup_kwargs)
            else:
                warmup_kwargs = self.config.forward_kwargs.copy()
                _ = backend.forward(self.sample_inputs, warmup_kwargs)

        if backend.config.task in TEXT_GENERATION_TASKS:
            LOGGER.info("\t+ Additional warmup for Text Generation")
            _ = backend.generate(self.sample_inputs, self.config.generate_kwargs)
        elif backend.config.task in IMAGE_DIFFUSION_TASKS:
            LOGGER.info("\t+ Additional warmup for Image Diffusion")
            _ = backend.call(self.sample_inputs, self.config.call_kwargs)

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
        LOGGER.info("\t+ Running Text Generation energy tracking")

        prefill_kwargs = self.config.generate_kwargs.copy()
        prefill_kwargs.update({"max_new_tokens": 1, "min_new_tokens": 1})

        with self.energy_tracker.track(file_prefix="prefill"):
            prefill_volume = 0
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.prefill(inputs, prefill_kwargs)
                print("input length: " + str(len(inputs["input_ids"].size(dim=1))))
                print("batch size: " + str(input_shapes["batch_size"]))
                try:
                    prefill_volume += len(inputs["input_ids"]) * input_shapes["batch_size"]
                except:
                    prefill_volume +=1
        prefill_energy = self.energy_tracker.get_energy()

        self.report.prefill.energy = prefill_energy
        self.report.prefill.efficiency = Efficiency.from_energy(
            prefill_energy, prefill_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )

        with self.energy_tracker.track(file_prefix="generate"):
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.generate(inputs, self.config.generate_kwargs)

        generate_energy = self.energy_tracker.get_energy()
        decode_energy = generate_energy - prefill_energy
        decode_volume = self.atomic_decode_volume * self.config.num_samples

        self.report.decode.energy = decode_energy
        self.report.decode.efficiency = Efficiency.from_energy(
            decode_energy, decode_volume, unit=TEXT_GENERATION_EFFICIENCY_UNIT
        )

    def run_image_diffusion_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running Image Diffusion energy tracking")

        with self.energy_tracker.track(file_prefix="call"):
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.call(inputs, self.config.call_kwargs)

        call_energy = self.energy_tracker.get_energy()
        call_volume = self.atomic_call_volume * self.config.num_samples

        self.report.call.energy = call_energy
        self.report.call.efficiency = Efficiency.from_energy(
            call_energy, call_volume, unit=IMAGE_DIFFUSION_EFFICIENCY_UNIT
        )

    def run_inference_energy_tracking(self, backend: Backend[BackendConfigT]):
        LOGGER.info("\t+ Running Inference energy tracking")

        with self.energy_tracker.track(file_prefix="forward"):
            for inputs in tqdm(self.dataloader):
                inputs = backend.prepare_inputs(inputs)
                _ = backend.forward(inputs, self.config.forward_kwargs)

        forward_energy = self.energy_tracker.get_energy()
        forward_volume = self.atomic_forward_volume * self.config.num_samples

        self.report.forward.energy = forward_energy
        self.report.forward.efficiency = Efficiency.from_energy(
            forward_energy, forward_volume, unit=INFERENCE_EFFICIENCY_UNIT
        )

    @property
    def atomic_forward_volume(self) -> int:  # in samples
        return self.config.input_shapes["batch_size"]

    @property
    def atomic_call_volume(self) -> int:  # in images
        if "prompt" in self.sample_inputs:
            return self.config.input_shapes["batch_size"] * self.config.call_kwargs["num_images_per_prompt"]
        else:
            return self.config.input_shapes["batch_size"]

    @property
    def atomic_prefill_volume(self) -> int:  # in tokens
        if "input_ids" in self.sample_inputs:
            # text conditioned generation (1 bos token or sequence_length tokens)
            return self.config.input_shapes["batch_size"] * self.config.input_shapes.get("sequence_length", 1)
        else:
            # image/audio conditioned generation (1 bos token)
            return self.config.input_shapes["batch_size"]

    @property
    def atomic_per_token_volume(self) -> int:  # in tokens
        return (
            self.config.input_shapes["batch_size"]
            * self.config.generate_kwargs["num_beams"]  # at each beam stage there are num_beams tokens generated
        )

    @property
    def atomic_decode_volume(self) -> int:  # in tokens
        return (
            self.config.input_shapes["batch_size"]
            * self.config.generate_kwargs["num_beams"]  # at each beam stage there are num_beams tokens generated
            * (self.config.generate_kwargs["max_new_tokens"] - 1)  # 1 token is generated during prefill
        )

    def get_report(self) -> InferenceReport:
        return self.report
