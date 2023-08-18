from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import os

import torch
from optimum.exporters import TasksManager
from optimum.onnxruntime import ORTOptimizer
from optimum.utils import DEFAULT_DUMMY_SHAPES
from transformers.utils import is_torch_available
from optimum.exporters.onnx.base import OnnxConfig
from optimum.utils.save_utils import maybe_save_preprocessors
from optimum.exporters.onnx.constants import UNPICKABLE_ARCHS
from optimum.utils import DEFAULT_DUMMY_SHAPES, ONNX_WEIGHTS_NAME
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from requests.exceptions import ConnectionError as RequestsConnectionError
from optimum.exporters.error_utils import AtolError, OutputMatchError, ShapeError
from optimum.exporters.onnx.convert import export_models, validate_models_outputs
from optimum.exporters.onnx.__main__ import logger, _get_submodels_and_onnx_configs
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
    ImageProcessingMixin,
    FeatureExtractionMixin,
    ProcessorMixin,
)
from optimum.exporters.onnx import (
    get_encoder_decoder_models_for_export,
    get_decoder_models_for_export,
    OnnxConfigWithPast,
    export_models,
)


def normalize_model_shapes(shapes: Dict[str, Any]) -> Dict[str, int]:
    vocab_size = shapes.get("vocab_size", None)  # text input
    num_labels = shapes.get("num_labels", None)  # classification labels
    num_queries = shapes.get("num_queries", None)  # object detection labels
    num_channels = shapes.get("num_channels", None)  # image input
    image_size = shapes.get("image_size", None)  # image input

    if image_size is None:
        image_size = shapes.get("size", None)  # image input

    if type(image_size) in [int, float]:
        height = image_size
        width = image_size
    elif type(image_size) in [list, tuple]:
        height = image_size[0]
        width = image_size[1]
    elif type(image_size) == dict:
        height = list(image_size.values())[0]
        width = list(image_size.values())[1]
    else:
        height = None
        width = None

    return {
        "vocab_size": vocab_size,
        "num_labels": num_labels,
        "num_queries": num_queries,
        "num_channels": num_channels,
        "height": height,
        "width": width,
    }


def get_model_shapes(
    config: Optional[PretrainedConfig] = None,
    preprocessor: Optional[
        Union[
            PreTrainedTokenizer,
            ImageProcessingMixin,
            FeatureExtractionMixin,
            ProcessorMixin,
        ]
    ] = None,
) -> Dict[str, int]:
    model_shapes = {}

    if config is not None:
        config_dict = {k: v for k, v in config.to_dict().items() if v is not None}
        model_shapes.update(config_dict)

    if preprocessor is not None and hasattr(preprocessor, "to_dict"):
        preprocessor_dict = {
            k: v for k, v in preprocessor.to_dict().items() if v is not None
        }
        model_shapes.update(preprocessor_dict)

    model_shapes = normalize_model_shapes(model_shapes)

    return model_shapes


def randomize_weights(model):
    for param in model.parameters():
        if torch.cuda.is_available() and param.device.type == "cpu":
            # we take advantage of the fact that a cuda device
            # is available to use cuda kernels for randomization
            # this is slower than asynchronous randomization while
            # model is fully on gpu (because of data transfer) but
            # faster than randomization while model is on cpu
            param.data.cuda().normal_(mean=0.0, std=0.2).cpu()
        else:
            param.data.normal_(mean=0.0, std=0.2)


def format_ort_quantization_dict(quantization_dict: Dict[str, Any]) -> None:
    from onnxruntime.quantization import (
        QuantFormat,
        QuantizationMode,
        QuantType,
    )

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

    return quantization_dict


def quantize_dummy_model(
    model,
    bnb_quantization_config,
):
    from accelerate.utils.bnb import (
        get_keys_to_not_convert,
        replace_with_bnb_layers,
        logger,
    )

    # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
    if bnb_quantization_config.skip_modules is None:
        bnb_quantization_config.skip_modules = get_keys_to_not_convert(model)

    # add cpu modules to skip modules only for 4-bit modules
    modules_to_not_convert = bnb_quantization_config.skip_modules

    # We add the modules we want to keep in full precision
    if bnb_quantization_config.keep_in_fp32_modules is None:
        bnb_quantization_config.keep_in_fp32_modules = []
    keep_in_fp32_modules = bnb_quantization_config.keep_in_fp32_modules
    modules_to_not_convert.extend(keep_in_fp32_modules)

    # compatibility with peft
    model.is_loaded_in_4bit = bnb_quantization_config.load_in_4bit
    model.is_loaded_in_8bit = bnb_quantization_config.load_in_8bit

    # quantization of an already loaded model
    logger.warning(
        "It is not recommended to quantize a loaded model. "
        "The model should be instantiated under the `init_empty_weights` context manager."
    )
    model = replace_with_bnb_layers(
        model, bnb_quantization_config, modules_to_not_convert=modules_to_not_convert
    )
    # convert param to the right dtype
    dtype = bnb_quantization_config.torch_dtype
    for name, param in model.state_dict().items():
        if any(
            module_to_keep_in_fp32 in name
            for module_to_keep_in_fp32 in keep_in_fp32_modules
        ):
            param.to(torch.float32)
            if param.dtype != torch.float32:
                name = name.replace(".weight", "").replace(".bias", "")
                param = getattr(model, name, None)
                if param is not None:
                    param.to(torch.float32)
        elif torch.is_floating_point(param):
            param.to(dtype)

    return model


def export_dummy_model(
    automodel_class,
    pretrained_config: PretrainedConfig,
    output_dir: str,
    device: torch.device,
    torch_dtype: Optional[torch.dtype] = None,
    auto_optimization: Optional[str] = None,
    use_merged: Optional[bool] = None,
    **cache_kwargs,
):
    ########################################
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = automodel_class.from_config(
            config=pretrained_config,
            torch_dtype=torch_dtype,
            trust_remote_code=cache_kwargs.get("trust_remote_code", False),
        )
    model.to_empty(device=device)
    randomize_weights(model)
    ########################################

    input_shapes = {}
    original_task = "auto"
    output_path = Path(output_dir)

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
        if (
            original_task == "auto"
        ):  # Make -with-past the default if --task was not explicitely specified
            task = task + "-with-past"

    onnx_config_constructor = TasksManager.get_exporter_config_constructor(
        model=model, exporter="onnx", task=task
    )
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
    if model.config.is_encoder_decoder and task.startswith(
        (
            "text2text-generation",
            "automatic-speech-recognition",
            "image-to-text",
            "feature-extraction-with-past",
        )
    ):
        models_and_onnx_configs = get_encoder_decoder_models_for_export(
            model, onnx_config
        )

    elif task.startswith("text-generation"):
        models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
    else:
        models_and_onnx_configs = {"model": (model, onnx_config)}

    print("Attempting to export the model to ONNX...")
    _, __ = export_models(
        models_and_onnx_configs=models_and_onnx_configs,  # type: ignore
        opset=opset,  # type: ignore
        output_dir=output_path,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=str(device),
        dtype="fp16" if torch_dtype == torch.float16 else None,
    )
    print("Model successfully exported to ONNX.")

    if auto_optimization:
        print("Attempting to optimize the exported ONNX models...")
        if onnx_files_subpaths is None:
            onnx_files_subpaths = [
                key + ".onnx" for key in models_and_onnx_configs.keys()
            ]
        optimizer = ORTOptimizer.from_pretrained(
            output_path, file_names=onnx_files_subpaths
        )

        optimization_config = AutoOptimizationConfig.with_optimization_level(
            optimization_level=auto_optimization
        )

        optimizer.optimize(
            save_dir=output_path,
            optimization_config=optimization_config,
            file_suffix="",
        )
        print("ONNX models successfully optimized.")

    # post process is disabled in optimum ort api so you need to export models with cli
    # and then load them with ort api to reproduce the same results
    if use_merged:
        try:
            print("Attempting to merge the exported ONNX models...")
            (
                models_and_onnx_configs,
                onnx_files_subpaths,
            ) = onnx_config.post_process_exported_models(
                output_path, models_and_onnx_configs, onnx_files_subpaths
            )
            print("ONNX models successfully merged.")
        except Exception as e:
            raise Exception(
                f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
            )


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    task: str = "auto",
    opset: Optional[int] = None,
    device: str = "cpu",
    fp16: Optional[bool] = False,
    optimize: Optional[str] = None,
    monolith: bool = False,
    no_post_process: bool = False,
    framework: Optional[str] = None,
    atol: Optional[float] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    pad_token_id: Optional[int] = None,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    for_ort: bool = False,
    do_validation: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    custom_onnx_configs: Optional[Dict[str, OnnxConfig]] = None,
    fn_get_submodels: Optional[Callable] = None,
    use_subprocess: bool = False,
    ########################################
    model: Optional[PreTrainedModel] = None,
    ########################################
    **kwargs_shapes,
):
    if optimize == "O4" and device != "cuda":
        raise ValueError(
            "Requested O4 optimization, but this optimization requires to do the export on GPU."
            " Please pass the argument `--device cuda`."
        )

    if (framework == "tf" and fp16 is True) or not is_torch_available():
        raise ValueError("The --fp16 option is supported only for PyTorch.")

    if fp16 is True and device == "cpu":
        raise ValueError(
            "The --fp16 option is supported only when exporting on GPU. Please pass the option `--device cuda`."
        )

    output = Path(output)
    if not output.exists():
        output.mkdir(parents=True)

    if for_ort:
        logger.warning(
            "The option --for-ort was passed, but its behavior is now the default in the ONNX exporter"
            " and passing it is not required anymore."
        )

    original_task = task
    task = TasksManager.map_from_synonym(task)

    framework = TasksManager.determine_framework(
        model_name_or_path, subfolder=subfolder, framework=framework
    )

    # get the shapes to be used to generate dummy inputs
    input_shapes = {}
    for input_name in DEFAULT_DUMMY_SHAPES.keys():
        input_shapes[input_name] = (
            kwargs_shapes[input_name]
            if input_name in kwargs_shapes
            else DEFAULT_DUMMY_SHAPES[input_name]
        )

    torch_dtype = None if fp16 is False else torch.float16

    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except KeyError as e:
            raise KeyError(
                f"The task could not be automatically inferred. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    if model is None:
        model = TasksManager.get_model_from_task(
            task,
            model_name_or_path,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            framework=framework,
            torch_dtype=torch_dtype,
            device=device,
        )

    custom_architecture = False
    is_stable_diffusion = "stable-diffusion" in task
    model_type = (
        "stable-diffusion"
        if is_stable_diffusion
        else model.config.model_type.replace("_", "-")
    )

    if not is_stable_diffusion:
        if model_type in TasksManager._UNSUPPORTED_CLI_MODEL_TYPE:
            raise ValueError(
                f"{model_type} is not supported yet. Only {TasksManager._SUPPORTED_CLI_MODEL_TYPE} are supported. "
                f"If you want to support {model_type} please propose a PR or open up an issue."
            )
        if model.config.model_type.replace(
            "-", "_"
        ) not in TasksManager.get_supported_model_type_for_task(task, exporter="onnx"):
            custom_architecture = True

    # TODO: support onnx_config.py in the model repo
    if custom_architecture and custom_onnx_configs is None:
        raise ValueError(
            f"Trying to export a {model.config.model_type.replace('-', '_')} model, that is a custom or unsupported architecture for the task {task}, but no custom onnx configuration was passed as `custom_onnx_configs`. Please refer to https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#custom-export-of-transformers-models for an example on how to export custom models. For the task {task}, the Optimum ONNX exporter supports natively the architectures: {TasksManager.get_supported_model_type_for_task(task, exporter='onnx')}."
        )

    if custom_architecture and original_task == "auto":
        raise ValueError(
            f'Automatic task detection is not supported with custom architectures. Please specify the `task` argument. Suggestion: task="{task}" (or task="{task}-with-past" if the model is decoder-based and supports KV cache)'
        )

    if (
        not custom_architecture
        and not is_stable_diffusion
        and task + "-with-past"
        in TasksManager.get_supported_tasks_for_model_type(model_type, "onnx")
    ):
        if (
            original_task == "auto"
        ):  # Make -with-past the default if --task was not explicitely specified
            task = task + "-with-past"
        else:
            logger.info(
                f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
                f" if needed, please pass `--task {task}-with-past` to export using the past key values."
            )

    if task.endswith("-with-past") and monolith is True:
        task_non_past = task.replace("-with-past", "")
        raise ValueError(
            f"The task {task} is not compatible with the --monolith argument. Please either use"
            f" `--task {task_non_past} --monolith`, or `--task {task}` without the monolith argument."
        )

    if original_task == "auto":
        synonyms_for_task = sorted(TasksManager.synonyms_for_task(task))
        if synonyms_for_task:
            synonyms_for_task = ", ".join(synonyms_for_task)
            possible_synonyms = f" (possible synonyms are: {synonyms_for_task})"
        else:
            possible_synonyms = ""
        logger.info(f"Automatic task detection to {task}{possible_synonyms}.")

    onnx_config, models_and_onnx_configs = _get_submodels_and_onnx_configs(
        model=model,
        task=task,
        monolith=monolith,
        custom_onnx_configs=custom_onnx_configs
        if custom_onnx_configs is not None
        else {},
        custom_architecture=custom_architecture,
        fn_get_submodels=fn_get_submodels,
    )

    if not is_stable_diffusion:
        needs_pad_token_id = (
            isinstance(onnx_config, OnnxConfigWithPast)
            and getattr(model.config, "pad_token_id", None) is None
            and task in ["text-classification"]
        )
        if needs_pad_token_id:
            if pad_token_id is not None:
                model.config.pad_token_id = pad_token_id
            else:
                try:
                    tok = AutoTokenizer.from_pretrained(model_name_or_path)
                    model.config.pad_token_id = tok.pad_token_id
                except Exception:
                    raise ValueError(
                        "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
                    )

        # Ensure the requested opset is sufficient
        if opset is None:
            opset = onnx_config.DEFAULT_ONNX_OPSET

        if opset < onnx_config.DEFAULT_ONNX_OPSET:
            raise ValueError(
                f"Opset {opset} is not sufficient to export {model_type}. "
                f"At least {onnx_config.DEFAULT_ONNX_OPSET} is required."
            )
        if atol is None:
            atol = onnx_config.ATOL_FOR_VALIDATION
            if isinstance(atol, dict):
                atol = atol[task.replace("-with-past", "")]

        # Saving the model config and preprocessor as this is needed sometimes.
        model.config.save_pretrained(output)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output)
        maybe_save_preprocessors(model_name_or_path, output)

        if model.config.is_encoder_decoder and task.startswith("text-generation"):
            raise ValueError(
                f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
                f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
                f" referring to `optimum.exporters.tasks.TaskManager`'s `_TASKS_TO_AUTOMODELS`."
            )

        onnx_files_subpaths = None
    else:
        # save the subcomponent configuration
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]
            if hasattr(subcomponent, "save_config"):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, "config") and hasattr(
                subcomponent.config, "save_pretrained"
            ):
                subcomponent.config.save_pretrained(output / model_name)

        onnx_files_subpaths = [
            os.path.join(name_dir, ONNX_WEIGHTS_NAME)
            for name_dir in models_and_onnx_configs
        ]

        # Saving the additional components needed to perform inference.
        model.scheduler.save_pretrained(output.joinpath("scheduler"))

        feature_extractor = getattr(model, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output.joinpath("feature_extractor"))

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output.joinpath("tokenizer"))

        tokenizer_2 = getattr(model, "tokenizer_2", None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))

        model.save_config(output)

    _, onnx_outputs = export_models(
        models_and_onnx_configs=models_and_onnx_configs,
        opset=opset,
        output_dir=output,
        output_names=onnx_files_subpaths,
        input_shapes=input_shapes,
        device=device,
        dtype="fp16" if fp16 is True else None,
        model_kwargs=model_kwargs,
    )

    if optimize is not None:
        from optimum.onnxruntime.configuration import AutoOptimizationConfig
        from optimum.onnxruntime import ORTOptimizer

        if onnx_files_subpaths is None:
            onnx_files_subpaths = [
                key + ".onnx" for key in models_and_onnx_configs.keys()
            ]
        optimizer = ORTOptimizer.from_pretrained(output, file_names=onnx_files_subpaths)

        optimization_config = AutoOptimizationConfig.with_optimization_level(
            optimization_level=optimize
        )

        optimization_config.disable_shape_inference = True
        optimizer.optimize(
            save_dir=output, optimization_config=optimization_config, file_suffix=""
        )

    # Optionally post process the obtained ONNX file(s), for example to merge the decoder / decoder with past if any
    # TODO: treating stable diffusion separately is quite ugly
    if not no_post_process and not is_stable_diffusion:
        try:
            logger.info("Post-processing the exported models...")
            (
                models_and_onnx_configs,
                onnx_files_subpaths,
            ) = onnx_config.post_process_exported_models(
                output, models_and_onnx_configs, onnx_files_subpaths
            )
        except Exception as e:
            raise Exception(
                f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
            )

    if is_stable_diffusion:
        use_subprocess = False  # TODO: fix Can't pickle local object 'get_stable_diffusion_models_for_export.<locals>.<lambda>'
    elif model.config.model_type in UNPICKABLE_ARCHS:
        # Pickling is bugged for nn.utils.weight_norm: https://github.com/pytorch/pytorch/issues/102983
        # TODO: fix "Cowardly refusing to serialize non-leaf tensor" error for wav2vec2-conformer
        use_subprocess = False

    if do_validation is True:
        try:
            validate_models_outputs(
                models_and_onnx_configs=models_and_onnx_configs,
                onnx_named_outputs=onnx_outputs,
                atol=atol,
                output_dir=output,
                onnx_files_subpaths=onnx_files_subpaths,
                input_shapes=input_shapes,
                device=device,
                dtype=torch_dtype,
                use_subprocess=use_subprocess,
                model_kwargs=model_kwargs,
            )
            logger.info(
                f"The ONNX export succeeded and the exported model was saved at: {output.as_posix()}"
            )
        except ShapeError as e:
            raise e
        except AtolError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except OutputMatchError as e:
            logger.warning(
                f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
            )
        except Exception as e:
            raise Exception(
                f"An error occured during validation, but the model was saved nonetheless at {output.as_posix()}. Detailed error: {e}."
            )
