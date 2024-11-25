from datasets import load_dataset
from tqdm import tqdm

from ...backends.base import Backend, BackendConfigT
from ...benchmark.report import BenchmarkReport
from ...preprocessors.dataset_preprocessor import TASKS_TO_PREPROCESSORS
from ...task_utils import IMAGE_DIFFUSION_TASKS, TEXT_GENERATION_TASKS
from ...trackers.energy import Efficiency, EnergyTracker
from ..base import Scenario
from .config import EnergyStarConfig

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


PREPROCESS_EFFICIENCY_UNIT = "samples/kWh"
FORWARD_EFFICIENCY_UNIT = "samples/kWh"
PREFILL_EFFICIENCY_UNIT = "tokens/kWh"
DECODE_EFFICIENCY_UNIT = "tokens/kWh"
CALL_EFFICIENCY_UNIT = "images/kWh"


class EnergyStarScenario(Scenario[EnergyStarConfig]):
    NAME = "energy-star"

    def __init__(self, config: EnergyStarConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> BenchmarkReport:
        self.task = backend.config.task

        if self.task in TEXT_GENERATION_TASKS:
            self.logger.info("\t+ Updating Text Generation kwargs with default values")
            self.config.generate_kwargs = {**TEXT_GENERATION_DEFAULT_KWARGS, **self.config.generate_kwargs}
            self.prefill_kwargs = {**self.config.generate_kwargs, **TEXT_GENERATION_PREFILL_OVERRIDES}
            self.logger.info("\t+ Initializing Text Generation report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "prefill", "decode"]
            )
        elif self.task in IMAGE_DIFFUSION_TASKS:
            self.logger.info("\t+ Updating Image Diffusion kwargs with default values")
            self.config.call_kwargs = {**IMAGE_DIFFUSION_DEFAULT_KWARGS, **self.config.call_kwargs}
            self.logger.info("\t+ Initializing Image Diffusion report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "call"]
            )
        else:
            self.logger.info("\t+ Initializing Inference report")
            self.report = BenchmarkReport.from_list(
                targets=["load_dataset", "preprocess_dataset", "load_model", "forward"]
            )

        self.energy_tracker = EnergyTracker(
            backend=backend.config.name,
            device=backend.config.device,
            device_ids=backend.config.device_ids,
        )

        self.run_dataset_loading_energy_tracking()
        self.run_model_loading_energy_tracking(backend)
        self.run_dataset_preprocessing_energy_tracking(backend)

        self.logger.info("\t+ Preparing sample inputs for model warmup")
        self.raw_sample_inputs = self.dataset[: self.config.input_shapes["batch_size"]]
        self.prepared_sample_inputs = backend.prepare_inputs(self.raw_sample_inputs)

        if self.task in TEXT_GENERATION_TASKS:
            self.warmup_text_generation(backend)
            self.run_text_generation_energy_tracking(backend)
        elif self.task in IMAGE_DIFFUSION_TASKS:
            self.warmup_image_diffusion(backend)
            self.run_image_diffusion_energy_tracking(backend)
        else:
            self.warmup_inference(backend)
            self.run_inference_energy_tracking(backend)

        return self.report

    # Dataset loading tracking
    def run_dataset_loading_energy_tracking(self):
        self.logger.info("\t+ Running dataset loading energy tracking")

        with self.energy_tracker.track(file_prefix="load_dataset"):
            self.dataset = load_dataset(
                self.config.dataset_name, self.config.dataset_config, split=self.config.dataset_split
            )

        self.report.load_dataset.energy = self.energy_tracker.get_energy()

    # Dataset preprocessing tracking
    def run_dataset_preprocessing_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running dataset preprocessing energy tracking")

        with self.energy_tracker.track(file_prefix="preprocess_dataset"):
            self.dataset = TASKS_TO_PREPROCESSORS[self.task](
                dataset=self.dataset,
                scenario_config=self.config,
                pretrained_config=backend.pretrained_config,
                pretrained_processor=backend.pretrained_processor,
            )

        preprocess_energy = self.energy_tracker.get_energy()
        preprocess_volume = self.dataset_preprocess_volume

        self.report.preprocess_dataset.energy = preprocess_energy
        self.report.preprocess_dataset.efficiency = Efficiency.from_energy(
            preprocess_energy,
            preprocess_volume,
            unit=PREPROCESS_EFFICIENCY_UNIT,
        )

    # Model loading tracking
    def run_model_loading_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running model loading energy tracking")

        with self.energy_tracker.track(file_prefix="load_model"):
            backend.load()

        self.report.load_model.energy = self.energy_tracker.get_energy()

    # Text Generation warmup
    def warmup_text_generation(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Text Generation")
        backend.generate(self.prepared_sample_inputs, self.config.generate_kwargs)
        for _ in range(self.config.warmup_runs):
            backend.generate(
                self.prepared_sample_inputs, {**self.config.generate_kwargs, **TEXT_GENERATION_WARMUP_OVERRIDES}
            )

    # Image Diffusion warmup
    def warmup_image_diffusion(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Image Diffusion")
        backend.call(self.prepared_sample_inputs, self.config.call_kwargs)
        for _ in range(self.config.warmup_runs):
            backend.call(self.prepared_sample_inputs, {**self.config.call_kwargs, **IMAGE_DIFFUSION_WARMUP_OVERRIDES})

    # Inference warmup
    def warmup_inference(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Warming up backend for Inference")
        for _ in range(self.config.warmup_runs):
            backend.forward(self.prepared_sample_inputs, self.config.forward_kwargs)

    # Text Generation energy tracking
    def run_text_generation_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Text Generation energy tracking")

        with self.energy_tracker.track(file_prefix="prefill"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.prefill(inputs, self.prefill_kwargs)

        prefill_energy = self.energy_tracker.get_energy()
        prefill_volume = self.dataset_prefill_volume

        self.report.prefill.energy = prefill_energy
        self.report.prefill.efficiency = Efficiency.from_energy(
            prefill_energy, prefill_volume, unit=PREFILL_EFFICIENCY_UNIT
        )

        with self.energy_tracker.track(file_prefix="generate"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.generate(inputs, self.config.generate_kwargs)

        generate_energy = self.energy_tracker.get_energy()
        decode_energy = generate_energy - prefill_energy
        decode_volume = self.dataset_decode_volume

        self.report.decode.energy = decode_energy
        self.report.decode.efficiency = Efficiency.from_energy(
            decode_energy,
            decode_volume,
            unit=DECODE_EFFICIENCY_UNIT,
        )

    # Image Diffusion energy tracking
    def run_image_diffusion_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Image Diffusion energy tracking")

        with self.energy_tracker.track(file_prefix="call"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.call(inputs, self.config.call_kwargs)

        call_energy = self.energy_tracker.get_energy()
        call_volume = self.dataset_call_volume

        self.report.call.energy = call_energy
        self.report.call.efficiency = Efficiency.from_energy(
            call_energy,
            call_volume,
            unit=CALL_EFFICIENCY_UNIT,
        )

    # Inference energy tracking
    def run_inference_energy_tracking(self, backend: Backend[BackendConfigT]):
        self.logger.info("\t+ Running Inference energy tracking")

        with self.energy_tracker.track(file_prefix="forward"):
            for i in tqdm(range(0, self.config.num_samples, self.config.input_shapes["batch_size"])):
                inputs = backend.prepare_inputs(self.dataset[i : i + self.config.input_shapes["batch_size"]])
                backend.forward(inputs, self.config.forward_kwargs)

        forward_energy = self.energy_tracker.get_energy()
        forward_volume = self.dataset_forward_volume

        self.report.forward.energy = forward_energy
        self.report.forward.efficiency = Efficiency.from_energy(
            forward_energy,
            forward_volume,
            unit=FORWARD_EFFICIENCY_UNIT,
        )

    @property
    def dataset_preprocess_volume(self) -> int:  # in samples
        return self.config.num_samples

    @property
    def dataset_forward_volume(self) -> int:  # in samples
        return self.config.num_samples

    @property
    def dataset_prefill_volume(self) -> int:  # in tokens
        prefill_volume = 0

        for sample in self.dataset:
            if "input_ids" in sample.keys():
                # text/image-text/video-image-text conditioned generation
                prefill_volume += self.raw_sample_inputs["input_ids"].numel()
            else:
                # image/audio/other conditioned generation (1 bos token)
                prefill_volume += 1

        return prefill_volume

    @property
    def dataset_per_token_volume(self) -> int:  # in tokens
        return (
            self.config.num_samples
            * self.config.generate_kwargs["num_beams"]  # at each beam stage there are num_beams tokens generated
        )

    @property
    def dataset_decode_volume(self) -> int:  # in tokens
        return (
            self.config.num_samples
            * self.config.generate_kwargs["num_beams"]  # at each beam stage there are num_beams tokens generated
            * (self.config.generate_kwargs["max_new_tokens"] - 1)  # 1 token is generated during prefill
        )

    @property
    def dataset_call_volume(self) -> int:  # in images
        if self.task == "text-to-image":
            return self.config.num_samples * self.config.call_kwargs["num_images_per_prompt"]
        else:
            return self.config.num_samples
