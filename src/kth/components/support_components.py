import numpy as np
from   typing import Iterable
from   enum import Enum
from   dataclasses import dataclass
import asyncio
import networkx as nx


from   symaware.base.data import Identifier
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

from kth.pybullet_env.environment import Environment
from kth.pybullet_env.dynamical_model import DroneCf2xModel
from kth.components.support_components import Network, ControlMessageType, ControlMessage
from kth.data.data import (ControlMessage, ControlMessageType)
from kth.components.high_level_controller import STLController


class StateOnlyPerceptionSystem(PerceptionSystem):
    """
    Unfortunately, this perception system is very limited, and can only perceive the state of the agent itself.
    """

    def _compute(self) -> dict[Identifier, StateObservation]:
        """
        Discard the information about any other agent.
        Only return the information related to the agent itself.
        """
        return {self._agent_id: StateObservation(self._agent_id, self._env.get_agent_state(self.agent_id))}

    

# Event based transmitter : When available send the control message
class Transmitter(CommunicationSender[EventAsyncLoopLock]): 
    __LOGGER = get_logger(__name__, "Transmitter")

    def __init__(self, agent_id: Identifier,network:Network):
        super().__init__(agent_id, EventAsyncLoopLock())
        self._network = network

    @log(__LOGGER)
    def _send_communication_through_channel(self, message: Message):
        self.__LOGGER.info(
            "Agent %d: Sending the message %s to the agent %d", self.agent_id, message, message.receiver_id
        )
        if message.receiver_id in self._network.message_queue:
            self._network.message_queue[message.receiver_id].put_nowait(message)
        else:
            self.__LOGGER.warning("Agent %d: Message %s could not be sent because Agent %d is not in the network", self.agent_id, message,message.receiver_id)
    @property
    def network(self):
        return self._network


#  Check for messages
class Receiver(CommunicationReceiver[DefaultAsyncLoopLock]):
    __LOGGER = get_logger(__name__, "MyCommunicationReceiver")
    
    def __init__(self, agent_id: int,network:Network):
        super().__init__(agent_id, DefaultAsyncLoopLock())
        
        self._network = network
        self._network.message_queue.setdefault(agent_id, asyncio.Queue()) # add yourself to the network
        self._get_message_task: asyncio.Task
    
    @property
    def network(self):
        return self._network
    
    @log(__LOGGER)
    def _receive_communication_from_channel(self) -> Iterable[Message]:
        raise NotImplementedError("This communication system can only be used asynchronously")

    @log(__LOGGER)
    def _decode_message(self, messages: tuple[ControlMessage]) -> "np.ndarray | None":
        if len(messages) == 0 or messages[0].receiver_id != self.agent_id:
            return None
        
        return messages

    @log(__LOGGER)
    async def _async_receive_communication_from_channel(self) -> Iterable[Message]:
        self.__LOGGER.info("Agent %d: Waiting for a message", self.agent_id)
        try:
            self._get_message_task = asyncio.create_task(self._network.message_queue[self.agent_id].get()) # wait until the item is available in case empty
            message: Message = await self._get_message_task
            self.__LOGGER.info(
                "Agent %d: Received the message %s from agent %d", self.agent_id, message, message.sender_id
            )
            
            controller : "STLController"= self._agent.controllers[0] # get reference to the controller of the agent
            control_neighbours = controller._task_neighbours_id
            if not (message.sender_id in control_neighbours):
                return tuple()
            else :
                return (message,)
        except asyncio.CancelledError:
            self.__LOGGER.info("Agent %d: Stopping waiting for new messages", self.agent_id)
            return tuple()

    def _update(self, message: "ControlMessage | None"):
        """Here we get the reference to controller in the agent and we make the modifications that we need"""
        controller : "STLController" = self._agent.controllers[0] # get reference to the controller of the agent
        
        # 
        if message.type == ControlMessageType.BEST_IMPACT :
            controller._best_impact_from_leaders[message.sender_id] = message.value
        elif message.type == ControlMessageType.WORSE_IMPACT :
            controller._worse_impact_from_follower = message.value
       
            
        
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
        
        
        
        self._fx_dot = 0.
        self._fy_dot = 0.
        
        self._fx_prev = 0.
        self._fy_prev = 0.
        
        self._fx_dot_prev = 0.
        self._fy_dot_prev = 0.
        
        self._vx_ref        = 0.2
        self._vx_dot_error  = 0
        self._vx_int_error  = 0
        self._vx_error      = 0
        self._vx_error_prev = 0
        
        
        self._vy_ref        = 0
        self._vy_dot_error  = 0
        self._vy_int_error  = 0
        self._vy_error      = 0
        self._vy_error_prev = 0
        

        self._integration_interval = 10 # seconds
        self._rewind_integrator_max_iter = int(self._integration_interval/self.async_loop_lock.time_interval)
        
        
        self.h_ref       = altitude_ref
        self._h_dot_error = 0
        self._h_int_error = 0
        
        self._h_error      = 0
        self._h_error_prev = 0
        self._integration_interval = 10 # seconds
        self._rewind_integrator_max_iter = int(self._integration_interval/self.async_loop_lock.time_interval)
        
        
        self._Kd_h              = 0.05
        self._Kp_h              = 0.001
        self._Ki_h              = 0.0
        
        self._Kp_v = 3
        self._Kd_v = 0.001
        self._Ki_v = 1
        
    
    
    def on_new_reference(self, new_reference: np.ndarray):
        """
        Set the new reference for the controller.
        The reference is a 3D vector with the x, y, and z components of the velocity.
        """
        self._vx_ref = new_reference[0]
        self._vy_ref = new_reference[1]
        
        
    def initialise_component(
            self,
            agent: "Agent",
            initial_awareness_database: MultiAgentAwarenessVector,
            initial_knowledge_database: MultiAgentKnowledgeDatabase,
        ):
        
        self._dynamical_model = agent.model
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        
        
 
    def _compute(self) :
        """Simple PD controller to track a given velocity profile in the x-y component"""
        
        
        current_vx,current_vy   = self._agent.self_awareness.state[7:9]
        current_height   = self._agent.self_awareness.state[2]
       
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
        
        
        self._fx_dot = (fx - self._fx_prev)/self.async_loop_lock.time_interval
        self._fy_dot = (fy - self._fy_prev)/self.async_loop_lock.time_interval
        
        self._fy_dot_dot = (self._fy_dot - self._fy_dot_prev)/self.async_loop_lock.time_interval
        self._fx_dot_dot = (self._fx_dot - self._fx_dot_prev)/self.async_loop_lock.time_interval
        
        
        self._fx_dot_prev = self._fx_dot
        self._fy_dot_prev = self._fy_dot
        
        self._fx_prev = fx
        self._fy_prev = fy
        
        forces = np.array([fx,fy,self._fx_dot_dot,self._fy_dot_dot]) # !todo: nicely change the framwork to get forces instead of RPM

        return forces, TimeSeries() # the dynamical model step function was modified to accept forces
    
        


@dataclass(frozen=True)
class Network:
    communication_graph : nx.Graph # edge only present where you have communication
    task_network        : nx.Graph # edges only present if there is a task
    full_network        : nx.Graph # complete graph of the system (all to all edges)
    message_queue       : dict[Identifier, asyncio.Queue] = {} # Used to store the messages for each agent
    
    
