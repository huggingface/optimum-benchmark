import time
from dataclasses import asdict, dataclass
from json import dump, load
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import pandas as pd
from flatten_dict import flatten, unflatten
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from transformers.utils.hub import http_user_agent
from typing_extensions import Self

LOGGER = getLogger("hub_utils")
HF_API = HfApi(user_agent=http_user_agent())


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, obj, owner):
        return self.fget(owner)


@dataclass
class PushToHubMixin:
    """
    A Mixin to push artifacts to the Hugging Face Hub
    """

    # DICTIONARY/JSON API
    def to_dict(self, flat=False) -> Dict[str, Any]:
        data = asdict(self)

        if flat:
            data = flatten(data, reducer="dot")

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PushToHubMixin":
        return cls(**data)

    def save_json(self, path: Union[str, Path], flat: bool = False) -> None:
        with open(path, "w") as f:
            dump(self.to_dict(flat=flat), f, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> Self:
        with open(path, "r") as f:
            data = load(f)
        return cls.from_dict(data)

    # DATAFRAME/CSV API
    def to_dataframe(self) -> pd.DataFrame:
        flat_dict_data = self.to_dict(flat=True)
        return pd.DataFrame.from_dict(flat_dict_data, orient="index").T

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        data = df.to_dict(orient="records")[0]

        for k, v in data.items():
            if isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                # we correct lists that were converted to strings
                data[k] = eval(v)

            if v != v:
                # we correct nan to None
                data[k] = None

        data = unflatten(data, splitter="dot")
        return cls.from_dict(data)

    def save_csv(self, path: Union[str, Path]) -> None:
        self.to_dataframe().to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> Self:
        return cls.from_dataframe(pd.read_csv(path))

    # HUGGING FACE HUB API
    def push_to_hub(
        self, repo_id: str, filename: Optional[str] = None, subfolder: Optional[str] = None, **kwargs
    ) -> None:
        filename = str(filename or self.default_filename)
        subfolder = str(subfolder or self.default_subfolder)

        token = kwargs.pop("token", None)
        private = kwargs.pop("private", False)
        exist_ok = kwargs.pop("exist_ok", True)
        repo_type = kwargs.pop("repo_type", "dataset")

        HF_API.create_repo(repo_id, token=token, private=private, exist_ok=exist_ok, repo_type=repo_type)

        with TemporaryDirectory() as tmpdir:
            path_in_repo = (Path(subfolder) / filename).as_posix()
            path_or_fileobj = Path(tmpdir) / filename
            self.save_json(path_or_fileobj)
            HF_API.upload_file(
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                path_or_fileobj=path_or_fileobj,
                repo_type=repo_type,
                token=token,
                **kwargs,
            )

    @classmethod
    def from_pretrained(
        cls, repo_id: str, filename: Optional[str] = None, subfolder: Optional[str] = None, **kwargs
    ) -> Self:
        filename = str(filename or cls.default_filename)
        subfolder = str(subfolder or cls.default_subfolder)

        repo_type = kwargs.pop("repo_type", "dataset")

        try:
            resolved_file = HF_API.hf_hub_download(
                repo_id=repo_id, filename=filename, subfolder=subfolder, repo_type=repo_type, **kwargs
            )
        except HfHubHTTPError as e:
            LOGGER.warning("Error while downloading from Hugging Face Hub")
            if "Client Error: Too Many Requests for url" in str(e):
                LOGGER.warning("Client Error: Too Many Requests for url. Retrying in 15 seconds.")
                time.sleep(15)
                resolved_file = HF_API.hf_hub_download(
                    repo_id=repo_id, filename=filename, subfolder=subfolder, repo_type=repo_type, **kwargs
                )
            else:
                raise e
        config_dict = cls.from_json(resolved_file)

        return config_dict

    @classproperty
    def default_filename(self) -> str:
        return "file.json"

    @classproperty
    def default_subfolder(self) -> str:
        return "benchmarks"
