from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from logging import getLogger
from pandas import DataFrame


from optimum_benchmark.backends.base import Backend
from optimum_benchmark.benchmarks.base import Benchmark, BenchmarkConfig
from optimum_benchmark.generators.dummy_dataset import DummyDatasetGenerator


LOGGER = getLogger("training")

# resolvers
OmegaConf.register_new_resolver("is_cpu", lambda device: device == "cpu")


@dataclass
class TrainingConfig(BenchmarkConfig):
    name: str = "training"
    _target_: str = "optimum_benchmark.benchmarks.training.TrainingBenchmark"

    # dataset options
    dataset_shapes: DictConfig = DictConfig(
        {
            "dataset_size": 500,
            "sequence_length": 16,
        }
    )

    # training options
    training_arguments: DictConfig = DictConfig(
        {
            "output_dir": "./trainer_output",
            "skip_memory_metrics": False,
            "use_cpu": "${is_cpu:${device}}",
            "do_train": True,
            "do_eval": False,
            "do_predict": False,
            # add any other training arguments here
            ###### TrainingArguments ########
            # prediction_loss_only: bool = False,
            # per_device_train_batch_size: int = 8,
            # per_gpu_train_batch_size: int | None = None,
            # gradient_accumulation_steps: int = 1,
            # learning_rate: float = 0.00005,
            # weight_decay: float = 0,
            # adam_beta1: float = 0.9,
            # adam_beta2: float = 0.999,
            # adam_epsilon: float = 1e-8,
            # max_grad_norm: float = 1,
            # num_train_epochs: float = 3,
            # max_steps: int = -1,
            # lr_scheduler_type: SchedulerType | str = "linear",
            # warmup_ratio: float = 0,
            # warmup_steps: int = 0,
            # log_level: str | None = "passive",
            # log_level_replica: str | None = "warning",
            # log_on_each_node: bool = True,
            # logging_dir: str | None = None,
            # logging_strategy: IntervalStrategy | str = "steps",
            # logging_first_step: bool = False,
            # logging_steps: float = 500,
            # logging_nan_inf_filter: bool = True,
            # save_strategy: IntervalStrategy | str = "steps",
            # save_steps: float = 500,
            # save_total_limit: int | None = None,
            # save_safetensors: bool | None = False,
            # save_on_each_node: bool = False,
            # use_mps_device: bool = False,
            # seed: int = 42,
            # data_seed: int | None = None,
            # jit_mode_eval: bool = False,
            # use_ipex: bool = False,
            # bf16: bool = False,
            # fp16: bool = False,
            # fp16_opt_level: str = "O1",
            # half_precision_backend: str = "auto",
            # bf16_full_eval: bool = False,
            # fp16_full_eval: bool = False,
            # tf32: bool | None = None,
            # local_rank: int = -1,
            # ddp_backend: str | None = None,
            # tpu_num_cores: int | None = None,
            # tpu_metrics_debug: bool = False,
            # debug: str | List[DebugOption] = "",
            # dataloader_drop_last: bool = False,
            # eval_steps: float | None = None,
            # dataloader_num_workers: int = 0,
            # past_index: int = -1,
            # run_name: str | None = None,
            # disable_tqdm: bool | None = None,
            # remove_unused_columns: bool | None = True,
            # label_names: List[str] | None = None,
            # load_best_model_at_end: bool | None = False,
            # metric_for_best_model: str | None = None,
            # greater_is_better: bool | None = None,
            # ignore_data_skip: bool = False,
            # sharded_ddp: List[ShardedDDPOption] | str | None = "",
            # fsdp: List[FSDPOption] | str | None = "",
            # fsdp_min_num_params: int = 0,
            # fsdp_config: str | None = None,
            # fsdp_transformer_layer_cls_to_wrap: str | None = None,
            # deepspeed: str | None = None,
            # label_smoothing_factor: float = 0,
            # optim: OptimizerNames | str = default_optim,
            # optim_args: str | None = None,
            # adafactor: bool = False,
            # group_by_length: bool = False,
            # length_column_name: str | None = "length",
            # report_to: List[str] | None = None,
            # ddp_find_unused_parameters: bool | None = None,
            # ddp_bucket_cap_mb: int | None = None,
            # ddp_broadcast_buffers: bool | None = None,
            # dataloader_pin_memory: bool = True,
            # use_legacy_prediction_loop: bool = False,
            # push_to_hub: bool = False,
            # resume_from_checkpoint: str | None = None,
            # hub_model_id: str | None = None,
            # hub_strategy: HubStrategy | str = "every_save",
            # hub_token: str | None = None,
            # hub_private_repo: bool = False,
            # gradient_checkpointing: bool = False,
            # include_inputs_for_metrics: bool = False,
            # fp16_backend: str = "auto",
            # push_to_hub_model_id: str | None = None,
            # push_to_hub_organization: str | None = None,
            # push_to_hub_token: str | None = None,
            # mp_parameters: str = "",
            # auto_find_batch_size: bool = False,
            # full_determinism: bool = False,
            # torchdynamo: str | None = None,
            # ray_scope: str | None = "last",
            # ddp_timeout: int | None = 1800,
            # torch_compile: bool = False,
            # torch_compile_backend: str | None = None,
            # torch_compile_mode: str | None = None,
            # dispatch_batches: bool | None = None
        }
    )


class TrainingBenchmark(Benchmark):
    def __init__(self):
        super().__init__()

        self.training_throughput: float = 0
        self.training_runtime: float = 0

    def configure(self, config: TrainingConfig):
        super().configure(config)

        self.dataset_shapes = config.dataset_shapes

        self.training_arguments = config.training_arguments

        self.dummy_dataset_generator = DummyDatasetGenerator(
            dataset_shapes=self.dataset_shapes
        )

    def run(self, backend: Backend) -> None:
        LOGGER.info("Running training benchmark")

        training_dataset = self.dummy_dataset_generator.generate(
            task=backend.task,
            pretrained_config=backend.pretrained_config,
        )

        backend.prepare_for_training(
            training_dataset=training_dataset,
            training_arguments=self.training_arguments,
        )
        training_output = backend.train()

        print(training_output.metrics)
        self.training_throughput = training_output.metrics["train_samples_per_second"]
        self.training_runtime = training_output.metrics["train_runtime"]

    def get_results_df(self) -> DataFrame:
        results_dict = dict()

        results_dict["training_throughput"] = self.training_throughput
        results_dict["training_runtime"] = self.training_runtime

        return DataFrame(results_dict, index=[0])

    def save(self) -> None:
        LOGGER.info("Saving training results")
        results_df = self.get_results_df()
        results_df.to_csv("training_results.csv")
