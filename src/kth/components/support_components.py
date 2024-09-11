import numpy as np
from   typing import Iterable
from   enum import Enum
from   dataclasses import dataclass, field
import asyncio
import pybullet as p


from symaware.base.data import Identifier
from symaware.base.utils import NullObject, Tasynclooplock

from symaware.base import (
    Controller,
    KnowledgeDatabase,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    initialize_logger,
    log,
    Identifier,
    PerceptionSystem,
    StateObservation,
    EventAsyncLoopLock,
    DefaultAsyncLoopLock,
    InfoMessage,
    CommunicationSender,
    Message,
    CommunicationReceiver,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    Agent
)

from kth.data.data import (ControlMessage, ControlMessageType)
from kth.components.high_level_controller import STLController
from kth.pybullet_env.environment import CoordinatedClock


class PyBulletPerceptionSystem(PerceptionSystem):
    
    __LOGGER = get_logger(__name__, "StateOnlyPerceptionSystem")
    def _compute(self) -> dict[Identifier, StateObservation]:
        """
        Discard the information about any other agent.
        Only return the information related to the agent itself.
        """
        
        new_perception_info = {}
        for agent_id in self._env._agent_entities.keys() :
            new_perception_info[agent_id] = StateObservation(agent_id, self._env.get_agent_state(agent_id))
        
        
        return new_perception_info


class PyBulletCamera(PerceptionSystem):
    """
    Unfortunately, this perception system is very limited, and can only perceive the state of the agent itself.
    """
    __LOGGER = get_logger(__name__, "Camera")
    
    
     # Update the projection matrix (shared by both cameras)
    projection_matrix = p.computeProjectionMatrixFOV(
                                                    fov=80,
                                                    aspect=1.0,
                                                    nearVal=0.1,
                                                    farVal=100.0
                                                    )
    
    
    def _compute(self) -> dict[Identifier, StateObservation]:
        """
        Discard the information about any other agent.
        Only return the information related to the agent itself.
        """
        
        state = self._agent.self_awareness.state
        
        pos = state[:3]
        orientation = state[3:7]
        
        euler_angles = p.getEulerFromQuaternion(orientation)
        yaw   = euler_angles[2]
        pitch = euler_angles[1]
        roll  = euler_angles[0]
        
        
        view_matrix1 = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=pos,
        distance=10,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=2
        )
        
        # Capture the image from Camera 1
        width, height, rgbImg1, depthImg1, segImg1 = p.getCameraImage(
            width=640,
            height=480,
            viewMatrix=view_matrix1,
            projectionMatrix=self.projection_matrix
        )

        return dict()
    
    def update(self, *args, **kwargs):
        pass
    
    
    
    
    
    
    
    
    

        
        
        

    

# Event based transmitter : When available send the control message
class Transmitter(CommunicationSender[EventAsyncLoopLock]): 
    __LOGGER = get_logger(__name__, "Transmitter")

    def __init__(self, agent_id: Identifier):
        super().__init__(agent_id, EventAsyncLoopLock())


    @log(__LOGGER)
    def _send_communication_through_channel(self, message: Message):
        self.__LOGGER.info(
            "Agent %d: Sending the message %s to the agent %d", self.agent_id, message, message.receiver_id
        )
        if message.receiver_id in Receiver.message_queue:
            Receiver.message_queue[message.receiver_id].put_nowait(message)
        else:
            self.__LOGGER.warning("Agent %d: Message %s could not be sent because Agent %d is not in the network", self.agent_id, message,message.receiver_id)



#  Check for messages
class Receiver(CommunicationReceiver[DefaultAsyncLoopLock]):
    __LOGGER = get_logger(__name__, "MyCommunicationReceiver")
    message_queue: dict[Identifier, asyncio.Queue[InfoMessage]] = {}
    
    def __init__(self, agent_id: int):
        
        super().__init__(agent_id, DefaultAsyncLoopLock())
        self.message_queue.setdefault(agent_id, asyncio.Queue())
        self._get_message_task: asyncio.Task
    
    @property
    def network(self):
        return self._network
    
    @log(__LOGGER)
    def _receive_communication_from_channel(self) -> Iterable[Message]:
        raise NotImplementedError("This communication system can only be used asynchronously")

    @log(__LOGGER)
    def _decode_message(self, messages: tuple[ControlMessage]) -> "int | None":
        
        
        if len(messages) == 0:
            return 0
        
        high_level_controller  = None
        controllers            = self._agent.controllers
        for controller in controllers:
            if isinstance(controller, STLController):
                high_level_controller = controller
                break
        if high_level_controller is None:
            raise ValueError("The high level controller is not available. Make sure that an STLController is available in the agent")
        
        for message in messages:
            if message.type == ControlMessageType.BEST_IMPACT :
                high_level_controller._best_impact_from_leaders[message.sender_id] = message.value
            elif message.type == ControlMessageType.WORSE_IMPACT :
                high_level_controller._worse_impact_from_follower = message.value
        
        return 1

    @log(__LOGGER)
    async def _async_receive_communication_from_channel(self) -> Iterable[Message|None]:
        self.__LOGGER.info("Agent %d: Waiting for a message", self.agent_id)
        try:
            self._get_message_task = asyncio.create_task(Receiver.message_queue[self.agent_id].get()) # wait until the item is available in case empty
            message: Message = await self._get_message_task
            self.__LOGGER.info(
                "Agent %d: Received the message %s from agent %d", self.agent_id, message, message.sender_id
            )
            
            high_level_controller  = None
            controllers = self._agent.controllers
            for controller in controllers:
               if isinstance(controller, STLController):
                   high_level_controller = controller
                   break
            if high_level_controller is None:
                raise ValueError("The high level controller is not available. Make sure that an STLController is available in the agent")
               
               
            control_neighbours =  high_level_controller._task_neighbours_id
            self.__LOGGER.info("Agent %d: Control neighbours are %s", self.agent_id, control_neighbours)
            if not (message.sender_id in control_neighbours):
                return tuple()
            else :
                return (message,)
        except asyncio.CancelledError:
            self.__LOGGER.info("Agent %d: Stopping waiting for new messages", self.agent_id)
            return tuple()

    def _update(self, decoding_successful: int )-> None:
        """Here we get the reference to controller in the agent and we make the modifications that we need"""
        pass
            
       
            
        
    async def async_stop(self):
        self._get_message_task.cancel()
        return await super().async_stop()
    
    
    
    
class VelocityController(Controller):
    __LOGGER = get_logger(__name__, "SimpleController")
    def __init__(
        self,
        agent_id,
        async_loop_lock: TimeIntervalAsyncLoopLock | None = None,
        altitude_ref: float = 1,
    ):
        super().__init__(agent_id, async_loop_lock)
    
        # dynamics along the x and y axis
        # .
        # v_x = g*theta
        #   ..
        # theta = 1/I_y * tau_theta
        #
        # Hence
        # ...              
        #  v_x = g/I_y * tau_theta -> third order system
        
        # They are the same value for these two axes
        
        
        self._coordinated_clock   : CoordinatedClock
        self._last_reference_time : float
        
        self._fx_dot = 0.
        self._fy_dot = 0.
        
        self._fx_prev = 0.
        self._fy_prev = 0.
        
        self._fx_dot_prev = 0.
        self._fy_dot_prev = 0.
        
        self._vx_ref        = 0.
        self._vx_dot_error  = 0.
        self._vx_int_error  = 0.
        self._vx_error      = 0.
        self._vx_error_prev = 0.
        
        
        self._vy_ref        = 0.
        self._vy_dot_error  = 0.
        self._vy_int_error  = 0.
        self._vy_error      = 0.
        self._vy_error_prev = 0.
        
        
        self._new_reference_time_out = 0.1 # seconds
        self._integration_interval = 10 # seconds
        self._rewind_integrator_max_iter = int(self._integration_interval/self.async_loop_lock.time_interval)
        
        
        self.h_ref       = altitude_ref
        self._h_dot_error = 0.
        self._h_int_error = 0.
        
        self._h_error      = 0.
        self._h_error_prev = 0.
        self._integration_interval = 10. # seconds
        self._rewind_integrator_max_iter = int(self._integration_interval/self.async_loop_lock.time_interval)
        
        
        self._Kd_h              = 0.05
        self._Kp_h              = 0.001
        self._Ki_h              = 0.0
        
        self._Kp_v = 3
        self._Kd_v = 0.001
        self._Ki_v = 1 
    
    
    def on_new_reference_velocity(self, new_reference: np.ndarray):
        """
        Set the new reference for the controller.
        The reference is a 3D vector with the x, y, and z components of the velocity.
        """
        self._vx_ref = new_reference[0]
        self._vy_ref = new_reference[1]
        self._last_reference_time = self._coordinated_clock.current_time

    
    def initialise_component(
            self,
            agent: "Agent",
            initial_awareness_database: MultiAgentAwarenessVector,
            initial_knowledge_database: MultiAgentKnowledgeDatabase,
        ):
        
        try :
            self._coordinated_clock :CoordinatedClock = initial_knowledge_database[self._agent_id]["coordinated_clock"]
        except KeyError:
            raise ValueError("The coordinated clock is not available in the knowledge database. A coordinated clock is needed for the controller to work")
        self._last_reference_time = self._coordinated_clock.current_time
        self._dynamical_model   = agent.model
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        
        
 
    def _compute(self) :
        """Simple PD controller to track a given velocity profile in the x-y component"""
        
        current_vx,current_vy     = self._agent.self_awareness.state[3:5]
        current_height            = self._agent.self_awareness.state[2]
        time_since_last_reference = self._coordinated_clock.current_time - self._last_reference_time
        
        if time_since_last_reference > self._new_reference_time_out:
            self._vx_ref = 0
            self._vy_ref = 0
            self._last_reference_time = self._coordinated_clock.current_time()
            
       
        # velocity error computation
        self._vx_error      = current_vx           - self._vx_ref
        self._vx_dot_error  =  (self._vx_error     - self._vx_error_prev)/self.async_loop_lock.time_interval
        self._vx_int_error  =   self._vx_int_error + self._vx_error*self.async_loop_lock.time_interval
        self._vx_error_prev =   self._vx_error
        
        self._vy_error      = current_vy           - self._vy_ref
        self._vy_dot_error  =  (self._vy_error     - self._vy_error_prev)/self.async_loop_lock.time_interval
        self._vy_int_error  =   self._vy_int_error + self._vy_error*self.async_loop_lock.time_interval
        self._vy_error_prev =   self._vy_error
        
        # altitude error computation
        self._h_error     = (current_height - self.h_ref)
        self._h_dot_error = (self._h_error- self._h_error_prev)/self.async_loop_lock.time_interval
        
        # integral and memory updates
        self._h_int_error += self._h_error*self.async_loop_lock.time_interval
        self._h_error_prev = self._h_error 
        
        # Only this part is really used
        fx = -self._Kp_v * self._vx_error - self._Kd_v * self._vx_dot_error
        fy = -self._Kp_v * self._vy_error - self._Kd_v * self._vy_dot_error
        
        self.__LOGGER.info("Agent %d: Current control input is %f %f", self.agent_id, fx, fy)
        
        
        
        self._fx_dot = (fx - self._fx_prev)/self.async_loop_lock.time_interval
        self._fy_dot = (fy - self._fy_prev)/self.async_loop_lock.time_interval
        
        self._fy_dot_dot = (self._fy_dot - self._fy_dot_prev)/self.async_loop_lock.time_interval
        self._fx_dot_dot = (self._fx_dot - self._fx_dot_prev)/self.async_loop_lock.time_interval
        
        
        self._fx_dot_prev = self._fx_dot
        self._fy_dot_prev = self._fy_dot
        
        self._fx_prev = fx
        self._fy_prev = fy
        
        pitch = fx/self._dynamical_model.iyy
        roll  = fy/self._dynamical_model.ixx
        
        torque_z = 0.
        
        forces = np.array([fx,fy,torque_z]) 

        return forces, TimeSeries() # the dynamical model step function was modified to accept forces
    
        
