from logging import getLogger
from contextlib import ExitStack

from ..base import Benchmark
from .config import TrainingConfig
from .report import TrainingReport
from .callback import MeasurementCallback
from ...trackers.memory import MemoryTracker
from ...trackers.energy import EnergyTracker
from ...backends.base import Backend, BackendConfigT
from ...generators.dataset_generator import DatasetGenerator

from transformers import default_data_collator

LOGGER = getLogger("training")


class TrainingBenchmark(Benchmark[TrainingConfig]):
    NAME = "training"

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__(config)

    def run(self, backend: Backend[BackendConfigT]) -> None:
        LOGGER.info("\t+ Creating dataset generator")
        dataset_generator = DatasetGenerator(
            task=backend.config.task,
            model_shapes=backend.model_shapes,
            dataset_shapes=self.config.dataset_shapes,
        )

        LOGGER.info("\t+ Generating training dataset")
        training_dataset = dataset_generator()

        LOGGER.info("\t+ Initializing training report")
        self.report = TrainingReport(
            num_processes=1,  # this process
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            per_process_batch_size=self.config.training_arguments["per_device_train_batch_size"],
            gradient_accumulation_steps=self.config.training_arguments["gradient_accumulation_steps"],
        )

        training_callbackes = []
        if self.config.latency:
            LOGGER.info("\t+ Adding latency measuring callback")
            latency_callback = MeasurementCallback(device=backend.config.device, backend=backend.config.name)
            training_callbackes.append(latency_callback)

        training_trackers = []
        if self.config.memory:
            LOGGER.info("\t+ Adding memory tracking context manager")
            memory_tracker = MemoryTracker(
                device=backend.config.device, backend=backend.config.name, device_ids=backend.config.device_ids
            )
            training_trackers.append(memory_tracker.track())

        if self.config.energy:
            LOGGER.info("\t+ Adding energy tracking context manager")
            energy_tracker = EnergyTracker(device=backend.config.device, device_ids=backend.config.device_ids)
            training_trackers.append(energy_tracker.track())

        with ExitStack() as stack:
            for tracker in training_trackers:
                stack.enter_context(tracker)

            backend.train(
                training_dataset=training_dataset,
                training_callbacks=training_callbackes,
                training_data_collator=default_data_collator,
                training_arguments=self.config.training_arguments,
            )

        if self.config.latency:
            self.report.populate_latency(all_latencies_list=latency_callback.get_latencies_list())
            self.report.log_latency()

        if self.config.memory:
            self.report.populate_memory(all_memories_dict=memory_tracker.get_memories_dict())
            self.report.log_memory()

        if self.config.energy:
            self.report.populate_energy(all_energies_dict=energy_tracker.get_energies_dict())
            self.report.log_energy()

    def get_report(self) -> TrainingReport:
        return self.report
