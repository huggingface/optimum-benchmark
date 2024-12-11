from tempfile import TemporaryDirectory
from typing import Any, Dict

from llama_cpp import Llama

from ..base import Backend
from .config import LlamaCppConfig


class LlamaCppBackend(Backend[LlamaCppConfig]):
    NAME: str = "llama_cpp"

    pretrained_model: Llama

    def __init__(self, config: LlamaCppConfig) -> None:
        super().__init__(config)

    def load(self) -> None:
        self.logger.info("\t+ Creating backend temporary directory")
        self.tmpdir = TemporaryDirectory()
        self.logger.info("\t+ Loading pretrained model")
        self.load_model_from_pretrained()
        self.tmpdir.cleanup()

    def load_model_from_pretrained(self) -> None:
        """
        Load the pretrained model from the given model name (normally GGUF, GGML)
        """

        self.pretrained_model = Llama.from_pretrained(
            self.config.model,
            **self.llama_cpp_kwargs,
        )

    @property
    def llama_cpp_kwargs(self) -> Dict[str, Any]:
        return {
            "embedding": self.config.task == "feature-extraction",
            "filename": self.config.filename,
            "verbose": False,
            "echo": False,
        }

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.task == "text-generation":
            if inputs["input_ids"].shape[0] != 1:
                raise ValueError("Batch size must be 1 for Text Generation with llama-cpp-python")
            return {"tokens": inputs["input_ids"].squeeze(0).tolist()}
        elif self.config.task == "feature-extraction":
            return {"input": [self.pretrained_model.detokenize(x).decode("utf-8") for x in inputs["input_ids"]]}
        else:
            raise ValueError(f"Task {self.config.task} not supported by {self.NAME}")

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        self.pretrained_model.embed(**inputs)

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        generator = self.pretrained_model.generate(**inputs, reset=True)
        for _ in range(kwargs["max_new_tokens"]):
            next(generator)

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        generator = self.pretrained_model.generate(**inputs, reset=True)
        for _ in range(kwargs["max_new_tokens"]):
            next(generator)
