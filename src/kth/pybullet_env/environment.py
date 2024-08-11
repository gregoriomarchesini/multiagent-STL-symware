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
        return np.array(position + orientation + velocity + angular_velocity )

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