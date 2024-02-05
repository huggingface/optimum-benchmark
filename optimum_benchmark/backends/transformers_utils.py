import os
import threading
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from torch.nn.modules import Module
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index
from torch.nn.parallel.parallel_apply import get_a_var
from transformers import (
    FeatureExtractionMixin,
    ImageProcessingMixin,
    PreTrainedTokenizer,
    GenerationConfig,
    PretrainedConfig,
    ProcessorMixin,
    AutoProcessor,
    AutoConfig,
)

PretrainedProcessor = Union[
    FeatureExtractionMixin,
    ImageProcessingMixin,
    PreTrainedTokenizer,
    ProcessorMixin,
]


def get_transformers_cache_dir():
    return os.path.expanduser("~/.cache/huggingface/hub")


def get_transformers_generation_config(model: str, **kwargs: Dict[str, Any]):
    try:
        # sometimes contains information about the model's input shapes that are not available in the config
        return GenerationConfig.from_pretrained(model, **kwargs)
    except Exception:
        return None


def get_transformers_pretrained_config(model: str, **kwargs: Dict[str, Any]):
    try:
        # sometimes contains information about the model's input shapes that are not available in the config
        return AutoConfig.from_pretrained(model, **kwargs)
    except ValueError:
        return None


def get_transformers_pretrained_processor(model: str, **kwargs: Dict[str, Any]):
    try:
        # sometimes contains information about the model's input shapes that are not available in the config
        return AutoProcessor.from_pretrained(model, **kwargs)
    except ValueError:
        return None


def extract_transformers_shapes_from_artifacts(
    config: PretrainedConfig, processor: Optional[PretrainedProcessor] = None
) -> Dict[str, Any]:
    shapes = {}
    artifacts_dict = {}

    config_dict = {k: v for k, v in config.to_dict().items() if v is not None}
    artifacts_dict.update(config_dict)

    if processor is not None and hasattr(processor, "to_dict"):
        processor_dict = {k: v for k, v in processor.to_dict().items() if v is not None}
        artifacts_dict.update(processor_dict)

    # text input
    shapes["vocab_size"] = artifacts_dict.get("vocab_size", None)
    shapes["type_vocab_size"] = artifacts_dict.get("type_vocab_size", None)
    shapes["max_position_embeddings"] = artifacts_dict.get("max_position_embeddings", None)
    if shapes["max_position_embeddings"] is None:
        shapes["max_position_embeddings"] = artifacts_dict.get("n_positions", None)

    # image input
    shapes["num_channels"] = artifacts_dict.get("num_channels", None)
    if shapes["num_channels"] is None:
        # processors have different names for the number of channels
        shapes["num_channels"] = artifacts_dict.get("channels", None)

    image_size = artifacts_dict.get("image_size", None)
    if image_size is None:
        # processors have different names for the image size
        image_size = artifacts_dict.get("size", None)

    if isinstance(image_size, (int, float)):
        shapes["height"] = image_size
        shapes["width"] = image_size
    elif isinstance(image_size, (list, tuple)):
        shapes["height"] = image_size[0]
        shapes["width"] = image_size[0]
    elif isinstance(image_size, dict) and len(image_size) == 2:
        shapes["height"] = list(image_size.values())[0]
        shapes["width"] = list(image_size.values())[1]
    elif isinstance(image_size, dict) and len(image_size) == 1:
        shapes["height"] = list(image_size.values())[0]
        shapes["width"] = list(image_size.values())[0]
    else:
        shapes["height"] = None
        shapes["width"] = None

    input_size = artifacts_dict.get("input_size", None)
    if input_size is not None:
        shapes["num_channels"] = input_size[0]
        shapes["height"] = input_size[1]
        shapes["width"] = input_size[2]

    # classification labels
    id2label = artifacts_dict.get("id2label", None)
    if id2label is not None:
        shapes["num_labels"] = len(id2label)

    num_classes = artifacts_dict.get("num_classes", None)
    if num_classes is not None:
        shapes["num_labels"] = num_classes

    # object detection labels
    shapes["num_queries"] = artifacts_dict.get("num_queries", None)
    if shapes["num_queries"] == 0:
        shapes["num_queries"] = 2

    return shapes


def randomize_weights(model):
    for param in model.parameters():
        if param.data.dtype in (torch.float32, torch.float16, torch.bfloat16):
            if torch.cuda.is_available() and param.device.type == "cpu":
                param.data.cuda().normal_(mean=0.0, std=0.2).cpu()
            elif torch.backends.mps.is_available() and param.device.type == "cpu":
                param.data.mps_normal_(mean=0.0, std=0.2)
            else:
                param.data.normal_(mean=0.0, std=0.2)
        elif param.data.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            if torch.cuda.is_available() and param.device.type == "cpu":
                param.data.cuda().randint_(low=-127, high=127).cpu()
            elif torch.backends.mps.is_available() and param.device.type == "cpu":
                param.data.mps_randint_(low=-127, high=127)
            else:
                param.data.randint_(low=-127, high=127)


# adapted from torch to use generate instead of forward
def parallel_generate_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Optional[Sequence[Dict[str, Any]]] = None,
    devices: Optional[Sequence[Optional[Union[int, torch.device]]]] = None,
) -> List[Any]:
    assert len(modules) == len(
        inputs
    ), f"The number of modules {len(modules)} is not equal to the number of inputs {len(inputs)}"
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = (cast(Dict[str, Any], {}),) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.cuda.current_stream(x) for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = (
        torch.is_grad_enabled(),
        torch.is_autocast_enabled(),
    )

    def _worker(
        i: int,
        module: Module,
        input: Any,
        kwargs: Dict[str, Any],
        device: Optional[Union[int, torch.device]] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            t = get_a_var(input)
            if t is None:
                with lock:
                    results[i] = ExceptionWrapper(
                        where=f"in replica {i}, no device was provided and no tensor input was found; "
                        "device cannot be resolved"
                    )
                return
            device = t.get_device()
        if stream is None:
            stream = torch.cuda.current_stream(device)
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module.generate(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where=f"in replica {i} on device {device}")

    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input, kwargs, device, stream))
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams)
            )
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


# adapted from torch to support generate
class TransformersDataParallel(torch.nn.DataParallel):
    def generate(self, *inputs: Any, **kwargs: Any) -> Any:
        with torch.autograd.profiler.record_function("DataParallel.generate"):
            if not self.device_ids:
                return self.module.generate(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError(
                        "module must have its parameters and buffers "
                        f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                        f"them on device: {t.device}"
                    )

            inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not module_kwargs:
                inputs = ((),)
                module_kwargs = ({},)

            if len(self.device_ids) == 1:
                return self.module.generate(*inputs[0], **module_kwargs[0])

            replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
            outputs = self.parallel_generate_apply(replicas, inputs, module_kwargs)
            return self.gather(outputs, self.output_device)

    def parallel_generate_apply(self, replicas: Sequence, inputs: Sequence, kwargs: Any) -> List[Any]:
        return parallel_generate_apply(replicas, inputs, kwargs, self.device_ids[: len(replicas)])

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
