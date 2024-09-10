from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class HardwareConfig:
    machine: str
    hardware: str
    subsets: List[str]
    backends: List[str]

    def __repr__(self):
        return (
            f"HardwareConfig(machine='{self.machine}', hardware='{self.hardware}', "
            f"subsets={self.subsets}, backends={self.backends})"
        )


def load_hardware_configs(file_path: str) -> List[HardwareConfig]:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return [HardwareConfig(**config) for config in data]
