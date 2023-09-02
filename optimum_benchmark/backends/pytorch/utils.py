import torch

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
