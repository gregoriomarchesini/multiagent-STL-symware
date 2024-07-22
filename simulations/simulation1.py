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

try:
    from symaware.simulators.pybullet import (
        DroneCf2pEntity,
        DroneCf2xEntity,
        DroneCf2xModel,
        DroneModel,
        DroneRacerEntity,
        DroneRacerModel,
        Environment,
    )
except ImportError as e:
    raise ImportError(
        "symaware-pybullet non found. "
        "Try running `pip install symaware-pybullet` or `pip install symaware[simulators]`"
    ) from e

#############################################################################################################################
# Define Multi-Agent System
#############################################################################################################################



























def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 0.01
    LOG_LEVEL = "INFO"
    MIN_DISTANCE_TO_EXPLODE = 1

    initialize_logger(LOG_LEVEL)

    entities: tuple[DroneCf2xEntity, DroneCf2pEntity, DroneRacerEntity] = (
        DroneCf2xEntity(0, model=DroneCf2xModel(0), position=np.array([1, 1, 2])),
        DroneRacerEntity(1, model=DroneRacerModel(1), position=np.array([0, 0, 2])),
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
        perception.add_on_computed(send_message_callback)
        agent.add_components(
            perception,
            MyController(agent.id, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL)),
            MyCommunicationSender(agent.id),
            MyCommunicationReceiver(agent.id, MIN_DISTANCE_TO_EXPLODE),
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












