from typing import Optional
from pathlib import Path
import torch


from optimum.exporters import TasksManager
from optimum.onnxruntime import ORTOptimizer
from optimum.utils import DEFAULT_DUMMY_SHAPES
from transformers import AutoTokenizer, PretrainedConfig
from optimum.utils.save_utils import maybe_save_preprocessors
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from optimum.exporters.onnx import (
    export_models,
    OnnxConfigWithPast,
    get_decoder_models_for_export,
    get_encoder_decoder_models_for_export,
)


def randomize_weights(model):
    for param in model.parameters():
        if param.dtype == torch.float32 or param.dtype == torch.float16:
            # float between -1 and 1 following the normal distribution
            param.data = torch.randn_like(param.data) * 0.05

        else:
            raise ValueError("Unsupported type")


def dummy_export(
    automodel_class,
    pretrained_config: PretrainedConfig,
    output_dir: str,
    device: torch.device,
    torch_dtype: Optional[torch.dtype],
    auto_optimization: Optional[str] = None,
    use_merged: Optional[bool] = None,
    **cache_kwargs,
):
    ########################################
    model = automodel_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=pretrained_config,
        state_dict={},
        torch_dtype=torch_dtype,
        device_map=device,
        **cache_kwargs,
    )
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
