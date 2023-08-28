from optimum.intel.openvino.utils import _HEAD_TO_AUTOMODELS

TASKS_TO_OVMODEL = {task: f"optimum.intel.openvino.{ovmodel}" for task, ovmodel in _HEAD_TO_AUTOMODELS.items()}
