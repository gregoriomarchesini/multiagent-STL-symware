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
        self._notify("stepping", self)
        for entity in self._agent_entities.values():
            entity.step()
        if not self.use_real_time:
            p.stepSimulation()
        
        self._coordinate_clock._step_time(self._sim_time_interval)
        self._notify("stepped", self)

    def stop(self):
        self._is_pybullet_initialized = False
        p.disconnect()
    
    def get_coordinated_clock(self):
        return self._coordinate_clock

    def set_debug_camera_position(
        self, distance: float, yaw: float, pitch: float, position: tuple[float, float, float]
    ):
        """
        Set the position of the debug camera in the pybullet environment.

        Args
        ----
        distance:
            Distance from the target
        yaw:
            Yaw angle of the camera
        pitch:
            Pitch angle of the camera
        position:
            Position of the camera
        """
        p.resetDebugVisualizerCamera(distance, yaw, pitch, position)

    def take_screenshot(
        self, width: int = 1440, height: int = 1120, shadow: bool = False, renderer: int = p.ER_TINY_RENDERER
    ) -> np.ndarray:
        """
        Take a screenshot of the current view of the camera and return it as a 3-dimensional numpy array
        of (height x width x rgba).
        The rgba values are in the interval [0, 1].

        An image produced this way can be saved on the disk using the matplotib utility.

        Example
        -------
        >>> # doctest: +SKIP
        >>> import pybullet as p
        >>> import matplotlib.pyplot as plt
        >>> from symaware.simulators.pybullet import Environment
        >>>
        >>> env = Environment(connection_method=p.DIRECT)
        >>> img = env.take_screenshot(width=1080, height=720)
        >>> plt.imsave("my_image", img)

        Args
        ----
        width:
            Width of the image
        height:
            Height of the image
        shadow:
            Whether to capture the shadows of the image
        renderer:
            Underlying renderer used by pybullet

        Returns:
            3-dimensional numpy array containing the screenshot of the simulation
        """
        _, _, img, _, _ = p.getCameraImage(width=width, height=height, shadow=shadow, renderer=renderer)
        return np.reshape(img, (height, width, 4)) * 1.0 / 255.0


class CoordinatedClock() :
    def __init__(self) -> None:
        self._clock_time = 0.0
    
    @property
    def current_time(self):
        return self._clock_time

    def _step_time(self, time_interval):
        self._clock_time += time_interval
        print(self._clock_time)