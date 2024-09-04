from typing import Any, Dict, List

import yaml


class HardwareConfig:
    def __init__(self, data: Dict[str, Any]):
        self.machine = data["machine"]
        self.description = data["description"]
        self.hardware_type = data["hardware_type"]
        self.subsets = data["subsets"]
        self.backends = data["backends"]

    def __repr__(self):
        return (
            f"HardwareConfig(machine='{self.machine}', description='{self.description}', "
            f"hardware_type={self.hardware_type}, subsets={self.subsets}, backends={self.backends})"
        )


def load_hardware_configs(file_path: str) -> List[HardwareConfig]:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return [HardwareConfig(config) for config in data]
