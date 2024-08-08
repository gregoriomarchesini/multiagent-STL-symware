import asyncio
from dataclasses import dataclass
from typing import Iterable
import casadi as ca

import numpy as np

from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    CommunicationReceiver,
    CommunicationSender,
    Controller,
    DefaultAsyncLoopLock,
    EventAsyncLoopLock,
    Identifier,
    InfoMessage,
    KnowledgeDatabase,
    Message,
    PerceptionSystem,
    StateObservation,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    initialize_logger,
    log,
)

from symaware.base.data import (
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
    TimeSeries,
)

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

class MyKnowledgeDatabase(KnowledgeDatabase):
    pass


class MyPerceptionSystem(PerceptionSystem):
    """
    Unfortunately, this perception system is very limited, and can only perceive the state of the agent itself.
    """

    def _compute(self) -> dict[Identifier, StateObservation]:
        """
        Discard the information about any other agent.
        Only return the information related to the agent itself.
        """
        #! Ernesto: non funziona
        return {self._agent_id: StateObservation(self._agent_id, self._env.get_agent_state(self.agent_id))}
    




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
        # .
        # theta = 1/I_y * tau_theta
        #
        # Hence
        #  ..
        #  v_x = g/I_y * tau_theta -> simple second order system with input tau_theta
        
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
        
    
    def initialise_component(
            self,
            agent: "Agent",
            initial_awareness_database: MultiAgentAwarenessVector,
            initial_knowledge_database: MultiAgentKnowledgeDatabase,
        ):
        
        self._dynamical_model = agent.model
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        
        try :
            I_yy = self._agent.model.iyy
            I_xx = self._agent.model.ixx
        except Exception as e:
            raise RuntimeError("The agent model does not have the inertia values. Please check the model of the agent to be a drone")
        
        
 
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
        
        
        
        
        print("current_vx",current_vx)
        print("current_vy",current_vy)
        print("target_vx",self._vx_ref)
        print("target_vy",self._vy_ref)
        print(self._fx_dot)
        model : DroneCf2xModel = self._agent.model
        
        forces = np.array([fx,fy,self._fx_dot_dot,self._fy_dot_dot]) # !todo: nicely change the framwork to get forces instead of RPM
        # check if you want to print that force is equal to gravity
        return forces, TimeSeries() # the dynamical model step function was modified to accept forces



        
def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 1./240.
    LOG_LEVEL = "INFO"

    initialize_logger(LOG_LEVEL)

    entities: tuple[DroneCf2xEntity, ...] = (
        DroneCf2xEntity(0, model=DroneCf2xModel(0), position=np.array([1, 1, 1])),
        DroneCf2xEntity(1, model=DroneCf2xModel(1), position=np.array([0, 0, 1])),
    )

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = Environment(async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
    # env.add_entities((DroneCf2pEntity(position=np.array([1, 1, 2])),))
    # env.add_entities((DroneCf2xEntity(position=np.array([2, 2, 2])),))
    # env.add_entities((DroneRacerEntity(position=np.array([0, 0, 2])),))

    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = AgentCoordinator[MyKnowledgeDatabase](env)
    for i, entity in enumerate(entities):
        ###########################################################
        # 2. Create the agent and assign it an entity             #
        ###########################################################
        agent = Agent[MyKnowledgeDatabase](i, entity)

        ###########################################################
        # 3. Add the agent to the environment                     #
        ###########################################################
        env.add_agents(agent)

        ###########################################################
        # 4. Create and set the component of the agent            #
        ###########################################################
        # In this example, all components run at the same frequency
        perception = MyPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        agent.add_components(
            perception,
            VelocityController(agent.id, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
        )

        ###########################################################
        # 5. Initialise the agent with some starting information  #
        ###########################################################
        agent.initialise_agent(AwarenessVector(agent.id, np.zeros(13)), {agent.id: MyKnowledgeDatabase()})

        ###########################################################
        # 6. Add the agent to the coordinator                     #
        ###########################################################
        agent_coordinator.add_agents(agent)

    ###########################################################
    # 7. Run the simulation                                   #
    ###########################################################
    # agent_coordinator.async_run()
    agent_coordinator.run(1/240.)


if __name__ == "__main__":
    main()
