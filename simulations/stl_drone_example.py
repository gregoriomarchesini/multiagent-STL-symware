
import numpy as np
import time 
from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    TimeIntervalAsyncLoopLock,
    initialize_logger,
)


from kth.pybullet_env.dynamical_model import DroneCf2xModel
from kth.pybullet_env.entities        import DroneCf2xEntity
from kth.pybullet_env.environment     import Environment
from kth.components.support_components import (VelocityController, 
                                               StateOnlyPerceptionSystem,
                                               Transmitter,
                                               Receiver)

from kth.components.high_level_controller import STLController
from kth.data.data import STLKnowledgeDatabase
from kth.stl.graphs import CommunicationGraph,TaskGraph, token_passing_algorithm
from kth.stl.stl import G,F,IndependentPredicate,CollaborativePredicate, regular_2D_polytope
from kth.stl.dynamics import SingleIntegrator2D


np.random.seed(100)

## Communication Graph.     
comm_graph = CommunicationGraph()
edges      = [(1,2)]
comm_graph.add_edges_from(edges)

# Create task graph
task_graph = TaskGraph()
# add tasks 

## Independent task : agent 1 goes toward position (5,0)
polytope     = regular_2D_polytope(4,  3)
predicate    = IndependentPredicate( polytope_0 = polytope, center = np.array([5.,0.]), agent_id =1 )
task         = G(10,20) @ predicate
task_graph.attach(task)

## Collaborative task : agent 2 stays close to agent 1
polytope     = regular_2D_polytope(6, 2)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([0.,0.]),source_agent_id =1, target_agent_id =2 )
task         = F(10,15) @ predicate
task_graph.attach(task)


leadership_tokens = token_passing_algorithm(task_graph, manually_set_leader=1)



def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 1./240.
    LOG_LEVEL = "INFO"

    initialize_logger(LOG_LEVEL)

    entities: tuple[DroneCf2xEntity, ...] = (
        DroneCf2xEntity(id=1, model=DroneCf2xModel(ID=1), position=np.array([1, 1, 1])),
        DroneCf2xEntity(id=2, model=DroneCf2xModel(ID=2), position=np.array([0, 0, 1])),
    )
    
    

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = Environment(async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))

    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = AgentCoordinator[STLKnowledgeDatabase](env)
       
    
    for i, entity in enumerate(entities):
        ###########################################################
        # 2. Create the agent and assign it an entity             #
        ###########################################################
        agent = Agent[STLKnowledgeDatabase](i, entity)

        ###########################################################
        # 3. Add the agent to the environment                     #
        ###########################################################
        env.add_agents(agent)

        ###########################################################
        # 4. Create and set the component of the agent            #
        ###########################################################
        # In this example, all components run at the same frequency
        mathematical_model     = SingleIntegrator2D(max_velocity = 1.8, unique_identifier = agent.id)
        
        
        perception_system      = StateOnlyPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        low_level_controller   = VelocityController(agent.id, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        high_level_controller  = STLController(agent.id,dynamical_model_math = mathematical_model, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        communication_sender   = Transmitter(agent.id,)
        communication_receiver = Receiver(agent.id) 
        
        high_level_controller.add("new_control_input",low_level_controller.on_new_reference)
        
        # create knowledge database
        tasks        = task_graph.task_list_for_node(agent.id)
        tokens       = leadership_tokens[agent.id]
        initial_time = 0. #! resolve time problem for all the system. When the mission starts they all need to have the same time (we should give a clock to the agent coordinato)
        
        knowledge_database = STLKnowledgeDatabase(stl_tasks=tasks, leadership_tokens=tokens, initial_time=initial_time)
        
        
        agent.add_components(
                             perception_system ,     
                             low_level_controller  , 
                             high_level_controller , 
                             communication_sender  , 
                             communication_receiver, 
        )

        ###########################################################
        # 5. Initialise the agent with some starting information  #
        ###########################################################
        agent.initialise_agent(AwarenessVector(agent.id, np.zeros(13)), {agent.id: knowledge_database})

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
