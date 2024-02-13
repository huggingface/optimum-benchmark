from dataclasses import dataclass, asdict
from typing import Union, Optional
from json import dump
import os

from transformers.configuration_utils import PushToHubMixin
from flatten_dict import flatten
import pandas as pd


@dataclass
class BenchmarkReport(PushToHubMixin):
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            kwargs["token"] = use_auth_token

        config_file_name = config_file_name if config_file_name is not None else "benchmark_report.json"

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)
        self.to_json(output_config_file)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_flat_dict(self) -> dict:
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
        return pd.DataFrame(flat_report_dict, index=[0])

    def to_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def log_all(self) -> None:
        raise NotImplementedError("`log_all` method must be implemented in the child class")
