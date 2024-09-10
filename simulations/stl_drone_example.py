
import numpy as np
import time 
import matplotlib.pyplot as plt
import os
from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    TimeIntervalAsyncLoopLock,
    initialize_logger,
    MultiAgentAwarenessVector
)


from kth.pybullet_env.dynamical_model import SingleIntegratorDroneModel
from kth.pybullet_env.entities        import DroneCf2xEntity,SphereEntity
from kth.pybullet_env.environment     import Environment
from kth.components.support_components import (VelocityController, 
                                               PyBulletPerceptionSystem,
                                               PyBulletCamera,
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
polytope     = regular_2D_polytope(5,  0.8)
predicate    = IndependentPredicate( polytope_0 = polytope, center = np.array([6.,6.]), agent_id =1 )
task         = G(10,20) @ predicate
task_graph.attach(task)


## Collaborative task 
polytope     = regular_2D_polytope(5, 0.8)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([0.,4.]),source_agent_id =1, target_agent_id =2 )
task         = (F(10,20)+ G(0,40)) @ predicate
task_graph.attach(task)

## Collaborative task 
polytope     = regular_2D_polytope(5, 0.8)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([4.,0.]),source_agent_id =1, target_agent_id =3 )
task         = (F(10,20) + G(0,40)) @ predicate
task_graph.attach(task)

## Collaborative task 
polytope     = regular_2D_polytope(5, 0.8)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([0.,-4.]),source_agent_id =1, target_agent_id =4 )
task         = (F(10,20) + G(0,40)) @ predicate
task_graph.attach(task)

## Collaborative task 
polytope     = regular_2D_polytope(5, 0.8)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([-4.,0.]),source_agent_id =1, target_agent_id =5 )
task         = (F(10,20)+ G(0,40)) @ predicate
task_graph.attach(task)

## Collaborative task 
polytope     = regular_2D_polytope(5, 0.8)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([2.,2.]),source_agent_id =2, target_agent_id =6 )
task         = (F(10,20)+ G(0,40)) @ predicate
task_graph.attach(task)

## Collaborative task 
polytope     = regular_2D_polytope(5, 0.8)
predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([2.,2.]),source_agent_id =3, target_agent_id =7 )
task         = (F(10,20)+ G(0,40)) @ predicate
task_graph.attach(task)

class Snapshot:

    counter: int = 0

    def save_snapshot(self, env: Environment):
        # Use this method to change the camera position.
        # It is possible to either set it at the very beginning or change it at every step
        env.set_debug_camera_position(10, 10, 0, (1, 2, 3))
        img = env.take_screenshot()

        os.makedirs("img", exist_ok=True)
        plt.imsave(f"img/frame-{self.counter}.jpeg", img)
        self.counter += 1


# ############################
# ## Collaborative task 
# polytope     = regular_2D_polytope(5, 1)
# predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([1.,1.]),source_agent_id =2, target_agent_id =3 )
# task         = F(40,50) @ predicate
# task_graph.attach(task)


# ## Collaborative task 
# polytope     = regular_2D_polytope(5, 1)
# predicate    = CollaborativePredicate( polytope_0 = polytope, center = np.array([2.,1.]),source_agent_id =7, target_agent_id =6 )
# task         = G(18,100) @ predicate
# task_graph.attach(task)




leadership_tokens    = token_passing_algorithm(task_graph, manually_set_leader=1)
print_tokens(leadership_tokens)
initial_agents_state = {1:np.array([0.,0.,1.,0.,0.,0.  , 0.,0.,0., 0.,0.,0.]),
                        2:np.array([-1.,0.8,1.,0.,0.,0., 0.,0.,0., 0.,0.,0.]), 
                        3:np.array([-0.2,0.,1.,0.,0.,0., 0.,0.,0., 0.,0.,0.]),
                        4:np.array([-0.5,0.5,1.,0.,0.,0, 0.,0.,0., 0.,0.,0.]),
                        5:np.array([1.,-0.8,1.,0.,0.,0., 0.,0.,0., 0.,0.,0.]),
                        6:np.array([1.,-2.8,1.,0.,0.,0., 0.,0.,0., 0.,0.,0.]),
                        7:np.array([-1.,1.8,1.,0.,0.,0., 0.,0.,0., 0.,0.,0.])}

max_velocity_agents = 5.
entities: tuple[DroneCf2xEntity, ...] = (  DroneCf2xEntity(id=agent_id, model=SingleIntegratorDroneModel(ID=agent_id), position=initial_agents_state[agent_id][:3]) for agent_id in initial_agents_state.keys())
target = SphereEntity(id=200, position=np.array([10.,10.,2.]), radius=1.0)
def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 1./240.
    LOG_LEVEL     = "ERROR"

    initialize_logger(LOG_LEVEL)
    snapshot = Snapshot()
    
    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = Environment(sim_time_interval = TIME_INTERVAL)
    env.add_on_stepped(snapshot.save_snapshot)
    coordinated_clock = env.get_coordinated_clock()

    ###########################################################
    # For each agent in the simulation...                     #
    ###########################################################
    agent_coordinator = AgentCoordinator[STLKnowledgeDatabase](env)
    env.add_entities(target)
    
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
        components             = []
        if agent.id == 1:
            components.append(PyBulletCamera(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL)))
            
        components.append(PyBulletPerceptionSystem(agent.id, env, TimeIntervalAsyncLoopLock(TIME_INTERVAL)))
        components.append(Transmitter(agent.id,))
        components.append(Receiver(agent.id))
        
        high_level_controller = STLController(agent.id,dynamical_model_math = mathematical_model, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        low_level_controller  =  VelocityController(agent.id, async_loop_lock=TimeIntervalAsyncLoopLock(TIME_INTERVAL))
        components.append(high_level_controller)
        components.append(low_level_controller)
        
        high_level_controller.add("new_control_input",low_level_controller.on_new_reference_velocity) # add callback function to the high level controller
        
        # create knowledge database
        tasks        = task_graph.task_list_for_node(agent.id)
        tokens       = leadership_tokens[agent.id]
        initial_time = coordinated_clock.current_time
        
        knowledge_database = STLKnowledgeDatabase(stl_tasks=tasks, leadership_tokens=tokens, initial_time=initial_time, coordinated_clock=coordinated_clock)
        
        
        agent.add_components(*components)

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
