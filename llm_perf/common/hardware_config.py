from typing import Any, Dict, List

import yaml


class HardwareConfig:
    def __init__(self, data: Dict[str, Any]):
        self.machine = data["machine"]
        self.description = data["description"]
        self.hardware_provider = data["hardware provider"]
        self.hardware_backend = data["hardware_backend type"]
        self.subsets = data["subsets"]
        self.backends = data["backends"]

    def __repr__(self):
        return (
            f"HardwareConfig(machine='{self.machine}', description='{self.description}', "
            f"hardware_type={self.hardware_backend}, subsets={self.subsets}, backends={self.backends})"
        )


def load_hardware_configs(file_path: str) -> List[HardwareConfig]:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return [HardwareConfig(config) for config in data]
