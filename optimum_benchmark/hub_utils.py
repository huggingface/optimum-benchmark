import os
import tempfile
from dataclasses import asdict, dataclass
from json import dump
from logging import getLogger
from typing import Any, Dict, Optional, Union

import pandas as pd
from flatten_dict import flatten
from huggingface_hub import create_repo, upload_file

LOGGER = getLogger(__name__)


@dataclass
class PushToHubMixin:
    """
    A Mixin to push artifacts to the Hugging Face Hub
    """

    def to_dict(self) -> Dict[str, Any]:
        data_dict = asdict(self)
        return data_dict

    def to_flat_dict(self) -> Dict[str, Any]:
        report_dict = self.to_dict()
        return flatten(report_dict, reducer="dot")

    def to_json(self, path: str, flat: bool = False) -> None:
        if flat:
            with open(path, "w") as f:
                dump(self.to_flat_dict(), f, indent=4)
        else:
            with open(path, "w") as f:
                dump(self.to_dict(), f, indent=4)

    def to_dataframe(self) -> pd.DataFrame:
        flat_report_dict = self.to_flat_dict()
        return pd.DataFrame.from_dict(flat_report_dict, orient="index")

    def to_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def save_pretrained(
        self,
        save_path: Optional[Union[str, os.PathLike]] = None,
        file_name: Optional[Union[str, os.PathLike]] = None,
        flat: bool = False,
    ) -> None:
        save_path = save_path or self.default_save_path
        file_name = file_name or self.default_file_name

        file_path = os.path.join(save_path, file_name)
        os.makedirs(save_path, exist_ok=True)
        self.to_json(file_path, flat=flat)

    def push_to_hub(
        self,
        repo_id: str,
        save_path: Optional[str] = None,
        file_name: Optional[str] = None,
        flat: bool = False,
        **kwargs,
    ) -> str:
        token = kwargs.get("token", None)
        private = kwargs.get("private", False)
        repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True).repo_id

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = save_path or self.default_save_path
            file_name = file_name or self.default_file_name

            path_or_fileobj = os.path.join(tmpdir, file_name)
            path_in_repo = os.path.join(save_path, file_name)
            self.to_json(path_or_fileobj, flat=flat)

            upload_file(
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                path_or_fileobj=path_or_fileobj,
                **kwargs,
            )

    @property
    def default_save_path(self) -> str:
        return "optimum-benchmark"

    @property
    def default_file_name(self) -> str:
        return "config.json"
