import os
import tempfile
from dataclasses import asdict
from json import dump
from logging import getLogger
from typing import Any, Dict, Optional, Union

import pandas as pd
from flatten_dict import flatten
from huggingface_hub import create_repo, upload_file

LOGGER = getLogger(__name__)


class PushToHubMixin:
    """
    A Mixin to push artifacts to the Hugging Face Hub
    """

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

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

    def push_to_hub(
        self,
        repo_id: str,
        file_name: Optional[Union[str, os.PathLike]] = None,
        path_in_repo: Optional[str] = None,
        flat: bool = False,
        **kwargs,
    ) -> str:
        token = kwargs.get("token", None)
        private = kwargs.get("private", False)
        repo_id = create_repo(repo_id, private=private, token=token, exist_ok=True).repo_id

        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = file_name or self.file_name
            path_or_fileobj = os.path.join(tmpdir, file_name)
            path_in_repo = path_in_repo or file_name
            self.to_json(path_or_fileobj, flat=flat)

            upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                **kwargs,
            )

    @property
    def file_name(self) -> str:
        return "config.json"
