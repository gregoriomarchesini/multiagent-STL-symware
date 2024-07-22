from kth.dynamical_model import (
        DroneCf2xModel,
        DroneModel,
        DroneRacerModel,
    )

from kth.entities import (
     DroneCf2pEntity,
     DroneCf2xEntity,
     DroneRacerEntity,

    )   
from kth.environment import (
     Environment,
)

import numpy as np
import pybullet as p
import math
import time
dt = 1./240.
import pybullet_data


p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0,0,-10)

drone = DroneCf2xEntity(0, model=DroneCf2xModel(0), position=np.array([1, 1, 2]))
drone.model.debug = True
drone.initialise()
drone.model._show_drone_local_axes()

radius=5
t = 0

while (1):
  t+=dt
  p.configureDebugVisualizer(lightPosition=[radius*math.sin(t),radius*math.cos(t),3])
  
  p.stepSimulation()
  time.sleep(dt)