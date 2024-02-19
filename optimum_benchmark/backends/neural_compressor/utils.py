from optimum.intel.neural_compressor.utils import _HEAD_TO_AUTOMODELS

TASKS_TO_INCMODELS = {
    task: f"optimum.intel.neural_compressor.{incmodel_name}" for task, incmodel_name in _HEAD_TO_AUTOMODELS.items()
}
