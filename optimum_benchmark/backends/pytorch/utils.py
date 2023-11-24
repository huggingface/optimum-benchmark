import threading
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import torch
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch.nn.modules import Module
from torch.nn.parallel.parallel_apply import get_a_var

DTYPES_MAPPING = {
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
}


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


# adapted from torch to use generate instead of forward
def parallel_generate_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Optional[Sequence[Dict[str, Any]]] = None,
    devices: Optional[Sequence[Optional[Union[int, torch.device]]]] = None,
) -> List[Any]:
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
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
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

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
