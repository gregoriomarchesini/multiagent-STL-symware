import os
from importlib import resources
from enum import Enum
import pybullet_data
import os

parent_dir = os.path.dirname(os.path.dirname(__file__))
os.path.join(parent_dir, 'assets')
class URDF(Enum):
    RACECAR = "racecar"
    DRONE_RACER = "drone_racer"
    DRONE_CF2P = "drone_cf2p"
    DRONE_CF2X = "drone_cf2x"

    @property
    def urdf(self) -> str:
        if self == URDF.RACECAR:
            return os.path.join(pybullet_data.getDataPath(), "racecar", "racecar_differential.urdf")
        elif self == URDF.DRONE_RACER:
            return str(os.path.join(parent_dir, 'assets', 'racer.urdf'))
        elif self == URDF.DRONE_CF2P:
            return  str(os.path.join(parent_dir, 'assets', 'cf2p.urdf'))
        elif self == URDF.DRONE_CF2X:
            return str(os.path.join(parent_dir, 'assets', 'cf2x.urdf'))
        else:
            raise ValueError(f"Unknown urdp {self}")
