import os
import gc
import shutil
from pathlib import Path
from logging import getLogger
from dataclasses import dataclass
from typing import Dict, List, Optional
from tempfile import TemporaryDirectory

import torch
import onnxruntime
from torch import Tensor
from omegaconf import OmegaConf
from accelerate import init_empty_weights
from optimum.exporters import TasksManager
from omegaconf.dictconfig import DictConfig
from optimum.utils import DEFAULT_DUMMY_SHAPES
from optimum.pipelines import ORT_SUPPORTED_TASKS
from transformers import AutoTokenizer, PreTrainedModel  # type: ignore
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.utils.save_utils import maybe_save_preprocessors
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from optimum.exporters.onnx import (
    export_models,
    OnnxConfigWithPast,
    get_decoder_models_for_export,
    get_encoder_decoder_models_for_export,
)
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    QuantizationConfig,
    AutoOptimizationConfig,
    AutoQuantizationConfig,
)


from src.backend.base import Backend, BackendConfig
from src.profiler.ort_profiler import ORTProfilingWrapper
from src.utils import infer_device_id

OmegaConf.register_new_resolver(
    "is_gpu",
    lambda device: torch.device(device).type == "cuda",
)
OmegaConf.register_new_resolver(
    "infer_provider",
    lambda device: f"{torch.device(device).type.upper()}ExecutionProvider",
)
OmegaConf.register_new_resolver(
    "infer_device_id", lambda device: infer_device_id(device)
)

LOGGER = getLogger("onnxruntime")


@dataclass
class ORTConfig(BackendConfig):
    name: str = "onnxruntime"
    version: str = onnxruntime.__version__
    _target_: str = "src.backend.onnxruntime.ORTBackend"

    # export options
    export: bool = True
    no_weights: bool = False
    use_merged: Optional[bool] = None
    torch_dtype: Optional[str] = None

    # provider options
    provider: str = "${infer_provider:${device}}"
    device_id: Optional[int] = "${infer_device_id:${device}}"  # type: ignore

    # inference options
    use_io_binding: bool = "${is_gpu:${device}}"  # type: ignore
    enable_profiling: bool = "${benchmark.profile}"  # type: ignore

    # optimization options
    optimization: bool = False
    optimization_config: DictConfig = DictConfig(
        {
            "optimization_level": 1,  # 0, 1, 2, 99
            "optimize_for_gpu": "${is_gpu:${device}}",
            "fp16": False,
            "enable_transformers_specific_optimizations": True,
            "enable_gelu_approximation": False,
            "disable_gelu_fusion": False,
            "disable_layer_norm_fusion": False,
            "disable_attention_fusion": False,
            "disable_skip_layer_norm_fusion": True,
            "disable_bias_skip_layer_norm_fusion": False,
            "disable_bias_gelu_fusion": False,
            "use_mask_index": False,
            "no_attention_mask": False,
            "disable_embed_layer_norm_fusion": True,
            "disable_shape_inference": False,
            "use_multi_head_attention": False,
            "enable_gemm_fast_gelu_fusion": False,
            "use_raw_attention_mask": False,
            "disable_group_norm_fusion": True,
            "disable_packed_kv": True,
        }
    )

    auto_optimization: Optional[str] = None  # O1, O2, O3, O4
    auto_optimization_config: DictConfig = DictConfig(
        {
            "for_gpu": "${is_gpu:${device}}",
            # add auto optimization specific options
        }
    )

    # quantization options
    quantization: bool = False
    quantization_config: DictConfig = DictConfig(
        {
            "is_static": False,
            "format": "QOperator",  # QOperator, QDQ
            "mode": "IntegerOps",  # QLinearOps, IntegerOps
            "activations_dtype": "QUInt8",  # QInt8, QUInt8
            "activations_symmetric": False,
            "weights_dtype": "QInt8",  # QInt8, QUInt8
            "weights_symmetric": True,
            "per_channel": False,
            "reduce_range": False,
            "operators_to_quantize": [
                "MatMul",
                "Add",
            ],
        }
    )

    # arm64,avx2,avx512,avx512_vnni,tensorrt
    auto_quantization: Optional[str] = None
    auto_quantization_config: DictConfig = DictConfig(
        {
            # for now, only dynamic quantization is supported
            "is_static": False
            # add auto quantization specific options
        }
    )

    # clean up options
    delete_cache: bool = False


class ORTBackend(Backend):
    def __init__(self, model: str, device: str, cache_kwargs: DictConfig) -> None:
        super().__init__(model, device, cache_kwargs)

        self.ortmodel_class = ORT_SUPPORTED_TASKS[self.task]["class"][0]
        self.automodel_class = TasksManager.get_model_class_for_task(
            task=self.task, framework="pt", model_type=self.pretrained_config.model_type
        )

        LOGGER.info(
            f"\t+ Infered AutoModel class {self.ortmodel_class.__name__} "
            f"for task {self.task} and model_type {self.pretrained_config.model_type}"
        )

    def configure(self, config: ORTConfig) -> None:
        super().configure(config)

        # session options
        self.session_options = onnxruntime.SessionOptions()
        if config.intra_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime session intra_op_num_threads({config.intra_op_num_threads})"
            )
            self.session_options.intra_op_num_threads = config.intra_op_num_threads
        if config.inter_op_num_threads is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime session inter_op_num_threads({config.inter_op_num_threads})"
            )
            self.session_options.inter_op_num_threads = config.inter_op_num_threads
        if config.enable_profiling:
            LOGGER.info("\t+ Enabling onnxruntime profiling")
            self.session_options.enable_profiling = True

        # provider options
        self.provider_options = {}
        if config.device_id is not None:
            LOGGER.info(
                f"\t+ Setting onnxruntime provider device_id({config.device_id})"
            )
            self.provider_options["device_id"] = config.device_id

        # clean up options
        if config.delete_cache:
            LOGGER.info("\t+ Will delete cache after benchmarking")
            self.delete_cache = True
        else:
            self.delete_cache = False

        # export options
        self.torch_dtype = getattr(
            torch, config.torch_dtype) if config.torch_dtype else None
        LOGGER.info(f"\t+ Using torch_dtype {self.torch_dtype}")

        with TemporaryDirectory() as tmpdirname:
            if config.no_weights:
                self.load_model_from_config(config, tmpdirname)
            else:
                self.load_model_from_pretrained(config)
                if config.optimization or config.auto_optimization is not None:
                    self.optimize_model(config, tmpdirname)

            if config.quantization or config.auto_quantization is not None:
                self.quantize_model(config, tmpdirname)

        # patch for ortmodel
        self.pretrained_model.device = self.device

    def load_model_from_config(self, config: ORTConfig, tmpdirname: str) -> None:
        LOGGER.info(
            "\t+ Creating empty weights model from config with on meta device")
        with init_empty_weights():
            self.pretrained_model = self.automodel_class.from_config(
                self.pretrained_config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.cache_kwargs.get(
                    "trust_remote_code", False),
            )
            self.pretrained_model.tie_weights()

        LOGGER.info(f"\t+ Materializing the model on {self.device}")
        self.pretrained_model = self.pretrained_model.to_empty(
            device=self.device)

        LOGGER.info(f"\t+ Randomizing weights")
        randomize_weights(self.pretrained_model)

        LOGGER.info(f"\t+ Exporting the model to ONNX")
        dummy_export(
            output_dir=tmpdirname,
            model=self.pretrained_model,
            device=self.device,
            torch_dtype=self.torch_dtype,
            auto_optimization=config.auto_optimization,
            use_merged=config.use_merged,
        )
        LOGGER.info(
            f"\t+ Created onnx files: {[f for f in os.listdir(tmpdirname) if f.endswith('.onnx')]}")

        LOGGER.info("\t+ Deleting pytorch model")
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading exported model in onnxruntime")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=Path(tmpdirname),
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
            use_merged=config.use_merged,
            export=False,
            **self.cache_kwargs,
        )

    def load_model_from_pretrained(self, config: ORTConfig) -> None:
        if self.torch_dtype != torch.float32:
            raise NotImplementedError(
                "Loading from pretrained is only supported with torch_dtype float32")

        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=self.model,
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
            use_merged=config.use_merged,
            export=config.export,
            **self.cache_kwargs,
        )

        LOGGER.info(
            f"\t+ Created onnx files: {[f for f in os.listdir(self.pretrained_model.model_save_dir) if f.endswith('.onnx')]}")

    def optimize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        if config.auto_optimization is not None:
            LOGGER.info(
                f"\t+ Using auto optimization {config.auto_optimization}")
            optimization_dict = OmegaConf.to_container(
                config.auto_optimization_config, resolve=True
            )
            LOGGER.info("\t+ Setting auto optimization parameters:")
            for key, value in optimization_dict.items():  # type: ignore
                LOGGER.info(f"\t\t+ {key}: {value}")

            optimization_config = AutoOptimizationConfig.with_optimization_level(
                optimization_level=config.auto_optimization, **optimization_dict
            )
        else:
            optimization_dict = OmegaConf.to_container(
                config.optimization_config, resolve=True
            )
            LOGGER.info("\t+ Setting optimization parameters:")
            for key, value in optimization_dict.items():  # type: ignore
                LOGGER.info(f"\t\t+ {key}: {value}")
            optimization_config = OptimizationConfig(**optimization_dict)

        LOGGER.info("\t+ Attempting optimization")
        optimizer = ORTOptimizer.from_pretrained(
            self.pretrained_model)
        optimizer.optimize(
            save_dir=f"{tmpdirname}/optimized",
            optimization_config=optimization_config,
        )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading optimized model")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/optimized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
        )

    def quantize_model(self, config: ORTConfig, tmpdirname: str) -> None:
        if config.auto_quantization is not None:
            LOGGER.info(
                f"\t+ Using auto quantization {config.auto_quantization}")
            auto_quantization_class = getattr(
                AutoQuantizationConfig, config.auto_quantization
            )
            quantization_dict = OmegaConf.to_container(
                config.auto_quantization_config, resolve=True
            )

            LOGGER.info("\t+ Setting quantization parameters:")
            for key, value in quantization_dict.items():  # type: ignore
                LOGGER.info(f"\t\t+ {key}: {value}")

            quantization_config = auto_quantization_class(**quantization_dict)

        else:
            quantization_dict = OmegaConf.to_container(
                config.quantization_config, resolve=True
            )
            # should be handeled by Pydantic later
            if quantization_dict.get("format", None) is not None:
                quantization_dict["format"] = QuantFormat.from_string(
                    quantization_dict["format"]
                )
            if quantization_dict.get("mode", None) is not None:
                quantization_dict["mode"] = QuantizationMode.from_string(
                    quantization_dict["mode"]
                )
            if quantization_dict.get("activations_dtype", None) is not None:
                quantization_dict["activations_dtype"] = QuantType.from_string(
                    quantization_dict["activations_dtype"]
                )
            if quantization_dict.get("weights_dtype", None) is not None:
                quantization_dict["weights_dtype"] = QuantType.from_string(
                    quantization_dict["weights_dtype"]
                )

            quantization_config = QuantizationConfig(
                **quantization_dict,
            )

        LOGGER.info("\t+ Attempting quantization")
        model_dir = self.pretrained_model.model_save_dir
        components = [file for file in os.listdir(
            model_dir) if file.endswith(".onnx")]
        for component in components:
            LOGGER.info(f"\t+ Quantizing {component}")
            quantizer = ORTQuantizer.from_pretrained(
                model_dir, file_name=component)
            quantizer.quantize(
                save_dir=f"{tmpdirname}/quantized",
                quantization_config=quantization_config,
            )
        self.delete_pretrained_model()

        LOGGER.info("\t+ Loading quantized model")
        self.pretrained_model = self.ortmodel_class.from_pretrained(
            model_id=f"{tmpdirname}/quantized",
            session_options=self.session_options,
            use_io_binding=config.use_io_binding,
            provider=config.provider,
            provider_options=self.provider_options,
        )

    def forward(self, input: Dict[str, Tensor]):
        output = self.pretrained_model(**input)
        return output

    def generate(self, input: Dict[str, Tensor], new_tokens: int) -> Tensor:
        output = self.pretrained_model.generate(
            **input,
            pad_token_id=self.pretrained_model.config.eos_token_id,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            num_beams=1,
        )
        return output

    def prepare_for_profiling(self, input_names: List[str]) -> None:
        LOGGER.info("Preparing for profiling")
        LOGGER.info("\t+ Wrapping model with profiler")
        self.pretrained_model = ORTProfilingWrapper(self.pretrained_model)

    def delete_pretrained_model(self) -> None:
        if hasattr(self, "pretrained_model"):
            del self.pretrained_model

        gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def delete_model_cache(self) -> None:
        model_cache_path = "models--" + self.model.replace("/", "--")
        model_cache_path = os.path.join(os.path.expanduser(
            "~/.cache/huggingface/hub"), model_cache_path)

        shutil.rmtree(model_cache_path, ignore_errors=True)

    def clean(self) -> None:
        LOGGER.info("Cleaning onnxruntime backend")
        self.delete_pretrained_model()

        if self.delete_cache:
            self.delete_model_cache()


def randomize_weights(model):
    for name, param in model.named_parameters():
        param.data = torch.rand_like(param.data) * 2 - 1

# a patch for random weights models exporting


def dummy_export(
    output_dir: str,
    model: PreTrainedModel,
    device: torch.device,
    torch_dtype: torch.dtype,
    auto_optimization: Optional[str] = None,
    use_merged: Optional[bool] = None,
):
    # matching cli behavior
    original_task = "auto"

    output_path = Path(output_dir)

    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = DEFAULT_DUMMY_SHAPES[input_name]

    try:
        task = TasksManager.infer_task_from_model(model)
    except KeyError as e:
        raise KeyError(
            f"The task could not be automatically inferred. Please provide the argument --task with the task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
        )

    if task + "-with-past" in TasksManager.get_supported_tasks_for_model_type(
        model.config.model_type.replace("_", "-"), "onnx"
    ):
        if original_task == "auto":  # Make -with-past the default if --task was not explicitely specified
            task = task + "-with-past"

    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="onnx", task=task)
    onnx_config = onnx_config_constructor(model.config)

    needs_pad_token_id = (
        isinstance(onnx_config, OnnxConfigWithPast)
        and getattr(model.config, "pad_token_id", None) is None
        and task in ["text-classification"]
    )
    if needs_pad_token_id:
        try:
            tok = AutoTokenizer.from_pretrained(model.name_or_path)
            model.config.pad_token_id = tok.pad_token_id
        except Exception:
            raise ValueError(
                "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
            )

    opset = onnx_config.DEFAULT_ONNX_OPSET
    atol = onnx_config.ATOL_FOR_VALIDATION
    if isinstance(atol, dict):
        atol = atol[task.replace("-with-past", "")]

    # Saving the model config and preprocessor as this is needed sometimes.
    model.config.save_pretrained(output_path)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_path)

    maybe_save_preprocessors(output_path, output_path)

    if model.config.is_encoder_decoder and task.startswith("text-generation"):
        raise ValueError(
            f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
            f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
            f" referring to `optimum.exporters.tasks.TaskManager`'s `_TASKS_TO_AUTOMODELS`."
        )

    onnx_files_subpaths = None
    if (
        model.config.is_encoder_decoder
        and task.startswith(
            (
                "text2text-generation",
                "automatic-speech-recognition",
                "image-to-text",
                "feature-extraction-with-past",
            )
        )
    ):
        models_and_onnx_configs = get_encoder_decoder_models_for_export(
            model, onnx_config)

    elif task.startswith("text-generation"):
        models_and_onnx_configs = get_decoder_models_for_export(
            model, onnx_config)
    else:
        models_and_onnx_configs = {"model": (model, onnx_config)}

    _, __ = export_models(
        models_and_onnx_configs=models_and_onnx_configs,  # type: ignore
        opset=opset,  # type: ignore
        output_dir=output_path,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=str(device),
        dtype="fp16" if torch_dtype == torch.float16 else None,
    )

    if auto_optimization:
        print("Attempting to optimize the exported ONNX models...")
        if onnx_files_subpaths is None:
            onnx_files_subpaths = [
                key + ".onnx" for key in models_and_onnx_configs.keys()]
        optimizer = ORTOptimizer.from_pretrained(
            output_path, file_names=onnx_files_subpaths)

        optimization_config = AutoOptimizationConfig.with_optimization_level(
            optimization_level=auto_optimization)

        optimizer.optimize(
            save_dir=output_path, optimization_config=optimization_config, file_suffix="")
        print("ONNX models successfully optimized.")

    # post process is disabled in optimum ort api so you need to export models with cli
    # and then load them with ort api to reproduce the same results
    if use_merged:
        try:
            print("Attempting to merge the exported ONNX models...")
            models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(
                output_path, models_and_onnx_configs, onnx_files_subpaths
            )
            print("ONNX models successfully merged.")
        except Exception as e:
            raise Exception(
                f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
            )
