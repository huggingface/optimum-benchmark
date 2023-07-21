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


def quantize_model(
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


def randomize_weights(model):
    for param in model.parameters():
        if torch.cuda.is_available() and param.device.type == "cpu":
            # we take advantage of the fact that a cuda device
            # is available to use cuda kernels for randomization
            # this is slower than randomization while model is
            # on gpu (because of data transfer) but faster than
            # randomization while model is on cpu
            param.data.cuda().normal_(mean=0.0, std=0.2).cpu()
        else:
            param.data.normal_(mean=0.0, std=0.2)


def dummy_main_export(
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


# def main_export(
#     model_name_or_path: str,
#     output: Union[str, Path],
#     task: str = "auto",
#     opset: Optional[int] = None,
#     device: str = "cpu",
#     fp16: Optional[bool] = False,
#     optimize: Optional[str] = None,
#     monolith: bool = False,
#     no_post_process: bool = False,
#     framework: Optional[str] = None,
#     atol: Optional[float] = None,
#     cache_dir: Optional[str] = None,
#     trust_remote_code: bool = False,
#     pad_token_id: Optional[int] = None,
#     subfolder: str = "",
#     revision: str = "main",
#     force_download: bool = False,
#     local_files_only: bool = False,
#     use_auth_token: Optional[Union[bool, str]] = None,
#     for_ort: bool = False,
#     do_validation: bool = True,
#     **kwargs_shapes,
# ):
#     output = Path(output)
#     if not output.exists():
#         output.mkdir(parents=True)

#     if for_ort:
#         logger.warning(
#             "The option --for-ort was passed, but its behavior is now the default in the ONNX exporter"
#             " and passing it is not required anymore."
#         )

#     original_task = task
#     task = TasksManager.map_from_synonym(task)

#     framework = TasksManager.determine_framework(model_name_or_path, subfolder=subfolder, framework=framework)

#     if (framework == "tf" and fp16 is True) or not is_torch_available():
#         raise ValueError("The --fp16 option is supported only for PyTorch.")

#     if fp16 is True and device == "cpu":
#         raise ValueError(
#             "The --fp16 option is supported only when exporting on GPU. Please pass the option `--device cuda`."
#         )

#     # get the shapes to be used to generate dummy inputs
#     input_shapes = {}
#     for input_name in DEFAULT_DUMMY_SHAPES.keys():
#         input_shapes[input_name] = (
#             kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
#         )

#     torch_dtype = None if fp16 is False else torch.float16
#     model = TasksManager.get_model_from_task(
#         task,
#         model_name_or_path,
#         subfolder=subfolder,
#         revision=revision,
#         cache_dir=cache_dir,
#         use_auth_token=use_auth_token,
#         local_files_only=local_files_only,
#         force_download=force_download,
#         trust_remote_code=trust_remote_code,
#         framework=framework,
#         torch_dtype=torch_dtype,
#     )

#     if task == "auto":
#         try:
#             task = TasksManager.infer_task_from_model(model_name_or_path)
#         except KeyError as e:
#             raise KeyError(
#                 f"The task could not be automatically inferred. Please provide the argument --task with the task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
#             )

#     if task != "stable-diffusion" and task + "-with-past" in TasksManager.get_supported_tasks_for_model_type(
#         model.config.model_type.replace("_", "-"), "onnx"
#     ):
#         if original_task == "auto":  # Make -with-past the default if --task was not explicitely specified
#             task = task + "-with-past"
#         else:
#             logger.info(
#                 f"The task `{task}` was manually specified, and past key values will not be reused in the decoding."
#                 f" if needed, please pass `--task {task}-with-past` to export using the past key values."
#             )

#     if task.endswith("-with-past") and monolith is True:
#         task_non_past = task.replace("-with-past", "")
#         raise ValueError(
#             f"The task {task} is not compatible with the --monolith argument. Please either use"
#             f" `--task {task_non_past} --monolith`, or `--task {task}` without the monolith argument."
#         )

#     if original_task == "auto":
#         synonyms_for_task = sorted(TasksManager.synonyms_for_task(task))
#         if synonyms_for_task:
#             synonyms_for_task = ", ".join(synonyms_for_task)
#             possible_synonyms = f" (possible synonyms are: {synonyms_for_task})"
#         else:
#             possible_synonyms = ""
#         logger.info(f"Automatic task detection to {task}{possible_synonyms}.")

#     if task != "stable-diffusion":
#         onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
#         onnx_config = onnx_config_constructor(model.config)

#         needs_pad_token_id = (
#             isinstance(onnx_config, OnnxConfigWithPast)
#             and getattr(model.config, "pad_token_id", None) is None
#             and task in ["text-classification"]
#         )
#         if needs_pad_token_id:
#             if pad_token_id is not None:
#                 model.config.pad_token_id = pad_token_id
#             else:
#                 try:
#                     tok = AutoTokenizer.from_pretrained(model_name_or_path)
#                     model.config.pad_token_id = tok.pad_token_id
#                 except Exception:
#                     raise ValueError(
#                         "Could not infer the pad token id, which is needed in this case, please provide it with the --pad_token_id argument"
#                     )

#         # Ensure the requested opset is sufficient
#         if opset is None:
#             opset = onnx_config.DEFAULT_ONNX_OPSET

#         if opset < onnx_config.DEFAULT_ONNX_OPSET:
#             raise ValueError(
#                 f"Opset {opset} is not sufficient to export {model.config.model_type}. "
#                 f"At least  {onnx_config.DEFAULT_ONNX_OPSET} is required."
#             )
#         if atol is None:
#             atol = onnx_config.ATOL_FOR_VALIDATION
#             if isinstance(atol, dict):
#                 atol = atol[task.replace("-with-past", "")]

#         # Saving the model config and preprocessor as this is needed sometimes.
#         model.config.save_pretrained(output)
#         generation_config = getattr(model, "generation_config", None)
#         if generation_config is not None:
#             generation_config.save_pretrained(output)
#         maybe_save_preprocessors(model_name_or_path, output)

#     if task == "stable-diffusion":
#         onnx_files_subpaths = [
#             "text_encoder/model.onnx",
#             "unet/model.onnx",
#             "vae_encoder/model.onnx",
#             "vae_decoder/model.onnx",
#         ]
#         models_and_onnx_configs = get_stable_diffusion_models_for_export(model)
#         # Saving the additional components needed to perform inference.
#         model.tokenizer.save_pretrained(output.joinpath("tokenizer"))
#         model.scheduler.save_pretrained(output.joinpath("scheduler"))
#         if model.feature_extractor is not None:
#             model.feature_extractor.save_pretrained(output.joinpath("feature_extractor"))
#         model.save_config(output)
#     else:
#         if model.config.is_encoder_decoder and task.startswith("text-generation"):
#             raise ValueError(
#                 f"model.config.is_encoder_decoder is True and task is `{task}`, which are incompatible. If the task was auto-inferred, please fill a bug report"
#                 f"at https://github.com/huggingface/optimum, if --task was explicitely passed, make sure you selected the right task for the model,"
#                 f" referring to `optimum.exporters.tasks.TaskManager`'s `_TASKS_TO_AUTOMODELS`."
#             )

#         onnx_files_subpaths = None
#         if (
#             model.config.is_encoder_decoder
#             and task.startswith(
#                 (
#                     "text2text-generation",
#                     "automatic-speech-recognition",
#                     "image-to-text",
#                     "feature-extraction-with-past",
#                 )
#             )
#             and not monolith
#         ):
#             models_and_onnx_configs = get_encoder_decoder_models_for_export(model, onnx_config)
#         elif task.startswith("text-generation") and not monolith:
#             models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
#         else:
#             models_and_onnx_configs = {"model": (model, onnx_config)}

#     _, onnx_outputs = export_models(
#         models_and_onnx_configs=models_and_onnx_configs,
#         opset=opset,
#         output_dir=output,
#         output_names=onnx_files_subpaths,
#         input_shapes=input_shapes,
#         device=device,
#         dtype="fp16" if fp16 is True else None,
#     )

#     if optimize == "O4" and device != "cuda":
#         raise ValueError(
#             "Requested O4 optimization, but this optimization requires to do the export on GPU."
#             " Please pass the argument `--device cuda`."
#         )

#     if optimize is not None:
#         from ...onnxruntime import AutoOptimizationConfig, ORTOptimizer

#         if onnx_files_subpaths is None:
#             onnx_files_subpaths = [key + ".onnx" for key in models_and_onnx_configs.keys()]
#         optimizer = ORTOptimizer.from_pretrained(output, file_names=onnx_files_subpaths)

#         optimization_config = AutoOptimizationConfig.with_optimization_level(optimization_level=optimize)

#         optimization_config.disable_shape_inference = True
#         optimizer.optimize(save_dir=output, optimization_config=optimization_config, file_suffix="")

#     # Optionally post process the obtained ONNX file(s), for example to merge the decoder / decoder with past if any
#     # TODO: treating stable diffusion separately is quite ugly
#     if not no_post_process and task != "stable-diffusion":
#         try:
#             logger.info("Post-processing the exported models...")
#             models_and_onnx_configs, onnx_files_subpaths = onnx_config.post_process_exported_models(
#                 output, models_and_onnx_configs, onnx_files_subpaths
#             )
#         except Exception as e:
#             raise Exception(
#                 f"The post-processing of the ONNX export failed. The export can still be performed by passing the option --no-post-process. Detailed error: {e}"
#             )

#     if do_validation is True:
#         try:
#             validate_models_outputs(
#                 models_and_onnx_configs=models_and_onnx_configs,
#                 onnx_named_outputs=onnx_outputs,
#                 atol=atol,
#                 output_dir=output,
#                 onnx_files_subpaths=onnx_files_subpaths,
#                 input_shapes=input_shapes,
#                 device=device,
#                 dtype=torch_dtype,
#             )
#             logger.info(f"The ONNX export succeeded and the exported model was saved at: {output.as_posix()}")
#         except ShapeError as e:
#             raise e
#         except AtolError as e:
#             logger.warning(
#                 f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
#             )
#         except OutputMatchError as e:
#             logger.warning(
#                 f"The ONNX export succeeded with the warning: {e}.\n The exported model was saved at: {output.as_posix()}"
#             )
#         except Exception as e:
#             raise Exception(
#                 f"An error occured during validation, but the model was saved nonetheless at {output.as_posix()}. Detailed error: {e}."
#             )
