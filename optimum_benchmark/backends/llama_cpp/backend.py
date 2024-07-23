from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

from llama_cpp import Llama

from optimum_benchmark.backends.base import Backend

from .config import LlamaCppConfig


class LlamaCppBackend(Backend[LlamaCppConfig]):
    NAME: str = "llama_cpp"

    def __init__(self, config: LlamaCppConfig) -> None:
        super().__init__(config)

        if self.config.no_weights:
            self.logger.info("\t+ Loading no weights model")
            raise NotImplementedError("No weights model is not yet implemented")

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
        embedding = True if self.config.task == "feature-extraction" else False

        self.pretrained_model = Llama.from_pretrained(
            repo_id=self.config.model,  # type: ignore
            filename=self.config.filename,
            verbose=False,
            echo=False,
            embedding=embedding
        )  # type: ignore

    def validate_task(self) -> None:
        if self.config.task not in ["text-generation"]:
            raise ValueError(f"Task {self.config.task} not supported by {self.NAME}")

    def prepare_inputs(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        if self.config.task == "text-generation":
            if inputs["input_ids"].shape[0] != 1:
                raise ValueError("Batch size must be 1 for Llama.cpp")

            inputs = super().prepare_inputs(inputs)
            inputs["tokens"] = inputs["input_ids"].squeeze()

            return inputs
        elif self.config.task == "feature-extraction":

            test = inputs["input_ids"].squeeze()
            print("test", test)
            print("inputs", self.pretrained_model.detokenize(test))
            hello = self.pretrained_model.detokenize(test)
            hello_str = hello.decode("utf-8")
            embeddings = self.pretrained_model.create_embedding(hello_str)
            print("embeddings", embeddings)

            raise NotImplementedError("Feature extraction not yet implemented")

        else:
            print("inputs", inputs)
            raise NotImplementedError(f"Quack")
        

    def forward(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        """
        Forward pass of the model\
        Evaluate the probabilites of the given tokens
        """
        return self.pretrained_model.eval(
            tokens=inputs["input_ids"],
        )

    def prefill(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        """
        Prefill the model with the input tokens
        We consider prefill as the time to first token, thus we evaluate the time it takes for the model to generate the first token
        """

        next(self.pretrained_model.generate(tokens=inputs["tokens"]))
        return inputs

    def generate(self, inputs: Dict[str, Any], kwargs: Dict[str, Any]) -> list[int]:
        """
        Generate new tokens from the pretrained model
        """

        output = []

        for token in self.pretrained_model.generate(tokens=inputs["tokens"]):
            output.append(token)
            if len(output) >= kwargs["max_new_tokens"]:
                break

        return output
