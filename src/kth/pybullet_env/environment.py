import os
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p
import pybullet_data
from symaware.base.models import Environment as BaseEnvironment
from symaware.base.utils import get_logger, log
from symaware.base import TimeIntervalAsyncLoopLock

from .entities import Entity

if TYPE_CHECKING:
    # String type hinting to support python 3.9
    from symaware.base.utils import AsyncLoopLock


class Environment(BaseEnvironment):
    """
    Environment based on the PyBullet physics engine.

    Args
    ----
    real_time_interval:
        If set to a strictly positive value, pybullet will run the simulation in real time.
        Otherwise, the simulation will run when :func:`step` is called.
    connection_method:
        Method used to connect to the pybullet server. See the pybullet documentation for more information.
    async_loop_lock:
        Async loop lock to use for the environment
    """

    __LOGGER = get_logger(__name__, "Pybullet.Environment")
    _clock_time = 0.0

    def __init__(
        self,
        sim_time_interval,
        real_time_interval: float = 0,
        connection_method: int = p.GUI,
    ):
        
        
        if sim_time_interval <= 0:
            raise ValueError(f"Expected a strictly positive sim_time_interval, got {sim_time_interval}")
        
        self._sim_time_interval = sim_time_interval
        async_loop_lock: "AsyncLoopLock | None" = TimeIntervalAsyncLoopLock(sim_time_interval) # this environment is only run at a fixed rate
        super().__init__(async_loop_lock)
        self._is_pybullet_initialized = False
        self._real_time_interval = real_time_interval
        self._connection_method = connection_method
        self._coordinate_clock = CoordinatedClock()

    @property
    def use_real_time(self) -> bool:
        return self._real_time_interval > 0

    @log(__LOGGER)
    def get_entity_state(self, entity: Entity) -> np.ndarray:
        if not isinstance(entity, Entity):
            raise TypeError(f"Expected SpatialEntity, got {type(entity)}")
        position, orientation      = p.getBasePositionAndOrientation(entity.entity_id)
        velocity, angular_velocity = p.getBaseVelocity(entity.entity_id)
        euler_angles               = p.getEulerFromQuaternion(orientation)
        return np.array(position + velocity + euler_angles + angular_velocity )

    @log(__LOGGER)
    def _add_entity(self, entity: Entity):
        if not isinstance(entity, Entity):
            raise TypeError(f"Expected SpatialEntity, got {type(entity)}")
        if not self._is_pybullet_initialized:
            self.initialise()
        entity.initialise()

    def initialise(self):
        if self._is_pybullet_initialized:
            return
        self._is_pybullet_initialized = True
        p.connect(self._connection_method)
        p.resetSimulation()
        p.setRealTimeSimulation(self.use_real_time)
        p.setGravity(0, 0, -9.81)
        p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))

    def step(self):
        for entity in self._agent_entities.values():
            entity.step()
        if not self.use_real_time:
            
        #     pos_agent_1 = self.get_entity_state(self._agent_entities[1])[:3]
        #     orientation = self.get_entity_state(self._agent_entities[1])[3:7]
            
        #     euler_angles = p.getEulerFromQuaternion(orientation)
        #     yaw   = euler_angles[2]
        #     pitch = euler_angles[1]
        #     roll  = euler_angles[0]
            
            
        #     view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=pos_agent_1,
        #     distance=10,
        #     yaw=yaw,
        #     pitch=pitch,
        #     roll=roll,
        #     upAxisIndex=2
        #     )


        #     # Update the projection matrix (shared by both cameras)
        #     projection_matrix = p.computeProjectionMatrixFOV(
        #         fov=60,
        #         aspect=1.0,
        #         nearVal=0.1,
        #         farVal=100.0
        #     )
            
        #     # Capture the image from Camera 1
        #     width, height, rgbImg1, depthImg1, segImg1 = p.getCameraImage(
        #         width=640,
        #         height=480,
        #         viewMatrix=view_matrix1,
        #         projectionMatrix=projection_matrix
        #     )
            
        #     rgbImg1 = np.reshape(rgbImg1, (height, width, 4))
            

    
            p.stepSimulation()
        
        self._coordinate_clock._step_time(self._sim_time_interval)

    def stop(self):
        self._is_pybullet_initialized = False
        p.disconnect()
    
    def get_coordinated_clock(self):
        return self._coordinate_clock
    



class CoordinatedClock() :
    def __init__(self) -> None:
        self._clock_time = 0.0
    
    @property
    def current_time(self):
        return self._clock_time

    def _step_time(self, time_interval):
        self._clock_time += time_interval
        print(self._clock_time)