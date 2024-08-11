
import numpy as np
import time 
from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    TimeIntervalAsyncLoopLock,
    initialize_logger,
    MultiAgentAwarenessVector
)


from kth.pybullet_env.dynamical_model import SingleIntegratorDroneModel
from kth.pybullet_env.entities        import DroneCf2xEntity
from kth.pybullet_env.environment     import Environment
from kth.components.support_components import (VelocityController, 
                                               StateOnlyPerceptionSystem,
                                               Transmitter,
                                               Receiver)

from kth.components.high_level_controller import STLController
from kth.data.data import STLKnowledgeDatabase
from kth.stl.graphs import CommunicationGraph,TaskGraph, token_passing_algorithm, print_tokens
from kth.stl.stl import G,F,IndependentPredicate,CollaborativePredicate, regular_2D_polytope
from kth.stl.dynamics import SingleIntegrator2D


np.random.seed(100)


# Create task graph
task_graph = TaskGraph()
# add tasks 

# Independent task 
polytope     = regular_2D_polytope(4,  1)
predicate    = IndependentPredicate( polytope_0 = polytope, center = np.array([4.,4.]), agent_id =1 )
task         = G(10,20) @ predicate
task_graph.attach(task)

polytope     = regular_2D_polytope(4,  2)
predicate    = IndependentPredicate( polytope_0 = polytope, center = np.array([8.,8.]), agent_id =1 )
task         = G(40,50) @ predicate
task_graph.attach(task)

## Collaborative task 
polytope     = regular_2D_polytope(6, 1)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([0.,0.]),source_agent_id =1, target_agent_id =2 )
task         = G(18,100) @ predicate
task_graph.attach(task)

## Collaborative task 
polytope     = regular_2D_polytope(6, 1)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([0.,0.]),source_agent_id =1, target_agent_id =3 )
task         = G(18,100) @ predicate
task_graph.attach(task)


## Collaborative task 
polytope     = regular_2D_polytope(6, 1)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([3.,1.]),source_agent_id =1, target_agent_id =4 )
task         = G(18,100) @ predicate
task_graph.attach(task)


## Collaborative task 
polytope     = regular_2D_polytope(6, 1)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-1.,3.]),source_agent_id =1, target_agent_id =5 )
task         = G(18,100) @ predicate
task_graph.attach(task)





leadership_tokens    = token_passing_algorithm(task_graph, manually_set_leader=1)
print_tokens(leadership_tokens)
initial_agents_state = {1:np.array([2.,2.,1.,0.,0.,0.]),
                        2:np.array([0.,0.,1.,0.,0.,0.]), 
                        3:np.array([4.,0.,1.,0.,0.,0.]),
                        4:np.array([0.,4.,1.,0.,0.,0.]),
                        5:np.array([4.,4.,1.,0.,0.,0.])}
max_velocity_agents = 5.
entities: tuple[DroneCf2xEntity, ...] = (  DroneCf2xEntity(id=agent_id, model=SingleIntegratorDroneModel(ID=agent_id), position=initial_agents_state[agent_id][:3]) for agent_id in initial_agents_state.keys())

def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 1./240.
    LOG_LEVEL = "ERROR"

    initialize_logger(LOG_LEVEL)
    
    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = Environment(sim_time_interval = TIME_INTERVAL)
    coordinated_clock = env.get_coordinated_clock()

    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = AgentCoordinator[STLKnowledgeDatabase](env)
       
    
    for i, entity in enumerate(entities, start=1):
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
        mathematical_model     = SingleIntegrator2D(max_velocity = max_velocity_agents, unique_identifier = agent.id)
        
        
        perception_system      = StateOnlyPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        high_level_controller  = STLController(agent.id,dynamical_model_math = mathematical_model, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        low_level_controller   = VelocityController(agent.id, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        communication_sender   = Transmitter(agent.id,)
        communication_receiver = Receiver(agent.id) 
        
        high_level_controller.add("new_control_input",low_level_controller.on_new_reference)
        
        # create knowledge database
        tasks        = task_graph.task_list_for_node(agent.id)
        tokens       = leadership_tokens[agent.id]
        initial_time = coordinated_clock.current_time #! resolve time problem for all the system. When the mission starts they all need to have the same time (we should give a clock to the agent coordinato)
        
        knowledge_database = STLKnowledgeDatabase(stl_tasks=tasks, leadership_tokens=tokens, initial_time=initial_time, coordinated_clock=coordinated_clock)
        
        
        agent.add_components(
                             perception_system ,     
                             high_level_controller , 
                             low_level_controller  , 
                             communication_sender  , 
                             communication_receiver, 
        )

        ###########################################################
        # 5. Initialise the agent with some starting information  #
        ###########################################################
        awareness_database : MultiAgentAwarenessVector = {agent_id: AwarenessVector(i, state) for agent_id,state in initial_agents_state.items()}
        agent.initialise_agent(awareness_database, {agent.id: knowledge_database})

        ###########################################################
        # 6. Add the agent to the coordinator                     #
        ###########################################################
        agent_coordinator.add_agents(agent)

    ###########################################################
    # 7. Run the simulation                                   #
    ###########################################################
    agent_coordinator.async_run()
    #agent_coordinator.run(TIME_INTERVAL)


if __name__ == "__main__":
    main()
