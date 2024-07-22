import asyncio
from dataclasses import dataclass
from typing import Iterable

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
    

class LowLevelController(Controller):
    __LOGGER = get_logger(__name__, "SimpleController")
    def __init__(
        self,
        agent_id,
        async_loop_lock: TimeIntervalAsyncLoopLock | None = None,
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
        
        
        self.altitude_ref = initial_awareness_database[self._agent_id].state[2]
        self._error_vx_prev = 0 # previous velocity error in vx
        self._error_vy_prev = 0 # previous velocity error in vy
        
        self.current_vx = 0
        self.current_vy = 0 
        
        self.current_target_vx = 0
        self.current_target_vy = 0
        
        # you can tune using matlab
        self._Kd = 0.1
        a        = I_xx/9.81 # gravity to inertia ratio
        zeta     = 0.9 # damping ration
        self._Kp = (self._Kd/2/np.sqrt(a)/zeta)**2
        
 
    def _compute(self) :
        """Simple PD controller to track a given velocity profile in the x-y component"""
        
        self._current_vx = 0
        self._current_vy = 0
        
        altitude_error = (self._agent.self_awareness.state[2] - self.altitude_ref)
        
        error_vx = self.current_target_vx  - self.current_vx
        error_vy = self.current_target_vy  - self.current_vy

        # check signs
        self.tau_theta = -self._Kp * error_vx - self._Kd * (error_vx - self._error_vx_prev)/self.async_loop_lock.time_interval
        self.tau_phi   = -self._Kp * error_vy - self._Kd * (error_vy - self._error_vy_prev)/self.async_loop_lock.time_interval
      
        self._error_vx_prev = error_vx
        self._error_vy_prev = error_vy
        model : DroneCf2xModel = self._agent.model
        
        torques = np.array([self.tau_phi,self.tau_theta, 0])
        force   = (4*model.hover_rpm**2 * model.kf) - (altitude_error)*0.001
        print(altitude_error)
        
        model : DroneCf2xModel = self._agent.model
        rpm = model.convert_force_and_torque_to_rpm(force,torques)
        # check if you want to print that force is equal to gravity
        return rpm, TimeSeries()



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
            LowLevelController(agent.id, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
        )

        ###########################################################
        # 5. Initialise the agent with some starting information  #
        ###########################################################
        agent.initialise_agent(AwarenessVector(agent.id, np.zeros(7)), {agent.id: MyKnowledgeDatabase()})

        ###########################################################
        # 6. Add the agent to the coordinator                     #
        ###########################################################
        agent_coordinator.add_agents(agent)

    ###########################################################
    # 7. Run the simulation                                   #
    ###########################################################
    agent_coordinator.async_run()


if __name__ == "__main__":
    main()
