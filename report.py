import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from flatten_dict import flatten
from flatten_dict.reducers import make_reducer


def gather_results(folder: Path):
    """
    Gather all results and configs from the given folder.

    Parameters
    ----------
    folder : Path
        The folder to search for results.

    Returns
    -------
    report : pd.DataFrame
        The report containing all results with their configurations in a single dataframe.
    """

    # List all csv results
    results_f = [f for f in folder.glob("**/perfs.csv")]
    configs_f = [f for f in folder.glob("**/config.yaml")]

    results_dfs = {
        f.relative_to(folder).parent.as_posix(): pd.read_csv(f, index_col=0)
        for f in results_f
    }

    configs_dfs = {
        f.relative_to(folder).parent.parent.as_posix(): pd.DataFrame(
            flatten(
                OmegaConf.load(f),
                reducer=make_reducer(delimiter='.')
            ), index=[0]
        )
        for f in configs_f
    }

    if len(results_dfs) == 0 or len(configs_dfs) == 0:
        raise ValueError(f"No results found in {folder}")

    # Merge perfs dataframes with configs
    reports = {
        name: configs_dfs[name].merge(
            results_dfs[name], left_index=True, right_index=True)
        for name in results_dfs.keys()
    }

    report = pd.concat(reports.values(), axis=0, ignore_index=True)

    # remove unnecessary columns
    report = report.drop(
        columns=[
            'backend._target_', 
            'experiment_name', 
            'experiment_datetime_id', 
            'python_version', 
            'transformers_version', 
            'optimum_version'
        ]
    )

    return report