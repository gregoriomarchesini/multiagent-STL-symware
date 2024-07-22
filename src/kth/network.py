import networkx as nx
from   dataclasses import dataclass
import numpy   as np
from   typing import Iterable
import asyncio


from   kth.dynamics import InputAffineDynamicalSymbolicModel
from   kth.signal_temporal_logic import StlTask,epsilon_position_closeness_predicate,AlwaysOperator,TimeInterval
from   symaware.base.data import Identifier



MANAGER = "manager"

@dataclass(frozen=True)
class Network:
    communication_graph : nx.Graph # edge only present where you have communication
    task_network        : nx.Graph # edges only present if there is a task
    full_network        : nx.Graph # complete graph of the system (all to all edges)
    message_queue       : dict[Identifier, asyncio.Queue] = {} # Used to store the messages for each agent
    
    def add_task_edge(self,edge:tuple[Identifier,Identifier]) -> None:
        
        if edge[0] not in self.task_network.nodes:
            raise Exception(f"Node {edge[0]} does not exist in the task network")
        if edge[1] not in self.task_network.nodes:
            raise Exception(f"Node {edge[1]} does not exist in the task network")
            
        if not self.task_network.has_edge(*edge):
            self.task_network.add_edge(*edge)
            self.task_network[edge[0]][edge[1]][MANAGER] = TaskManager(edge_tuple=edge)
        else :
            raise Exception(f"Edge {edge} already exists in the task network")
    
    def remove_task_edge(self,edge:tuple[Identifier,Identifier]) -> None:
        if self.task_network.has_edge(*edge):
            self.task_network.remove_edge(*edge)
    
    def add_communication_edge(self,edge:tuple[Identifier,Identifier]) -> None:
            
            if edge[0] not in self.communication_graph.nodes:
                raise Exception(f"Node {edge[0]} does not exist in the communication graph")
            if edge[1] not in self.communication_graph.nodes:
                raise Exception(f"Node {edge[1]} does not exist in the communication graph")
                
            if not self.communication_graph.has_edge(*edge):
                self.communication_graph.add_edge(*edge)
            else :
                raise Exception(f"Edge {edge} already exists in the communication graph")
    def remove_communication_edge(self,edge:tuple[Identifier,Identifier]) -> None:
        if self.communication_graph.has_edge(*edge):
            self.communication_graph.remove_edge(*edge)
    
    def clean(self)-> None :
        for edge in self.task_network.edges:
            if not self.task_network[edge[0]][edge[1]][MANAGER].has_tasks():
                self.task_network.remove_edge(*edge)

class TaskManager :
    """
    Data class to create a graph edge, This will contain all the barrier function defined over this edge.
    """
    def __init__(self,edge_tuple:tuple[Identifier,Identifier],weight = 1):
        if self.weight < 0:
            raise ValueError("Weight should be a positive number")

        self.edge_tuple             : tuple[Identifier,Identifier] = edge_tuple
        self.weight                 : float         = 1
        self.task_list              : list[StlTask] = []
      
    
    def _add_single_task(self, input_task: StlTask) -> None:
        """ Set the tasks for the edge that has to be respected by the edge. Input is expected to be a list  """
       
        if not isinstance(input_task, StlTask):
            raise Exception("please enter a valid STL task object or a list of StlTask objects")
        else:
            if isinstance(input_task, StlTask):
                # set the source node pairs of this node
                nodei,nodej = self.edge_tuple

                if (not nodei in input_task.contributing_agents) or (not nodej in input_task.contributing_agents) :
                    raise Exception(f"the task {input_task} is not compatible with the edge {self.edge_tuple}. The contributing agents are {input_task.contributing_agents}")
                else:
                    self.task_list.append(input_task) # adding a single task
            
    # check task addition
    def add_tasks(self, tasks: StlTask|list[StlTask]):
        if isinstance(tasks, list): # list of tasks
            for task in tasks:
                self._add_single_task(task)
        else: # single task case
            self._add_single_task(tasks)
    
    def clean_tasks(self)-> None :
        self.task_list      = []
    
    def has_tasks(self) -> bool:
        return len(self.task_list) > 0


def get_fully_connected_network(agents_identifiers: Iterable[Identifier]) :
    """Create a fully connected network with no tasks"""
    
    comm_graph = nx.complete_graph(agents_identifiers)
    task_graph = nx.complete_graph(agents_identifiers)
    
    for edge in task_graph.edges:
        task_graph[edge[0]][edge[1]][MANAGER] = TaskManager(edge_tuple=edge)
    
    full_network        = nx.complete_graph(agents_identifiers)
    
    return Network(communication_graph=comm_graph, task_network=task_graph, full_network=full_network)
    
    
def get_network_from_edges(communication_edges: list[tuple[Identifier, Identifier]], task_edges: list[tuple[Identifier, Identifier]]) -> nx.Graph:
    """
    Create a network from a list of communication edges and task edges.

    Args :
    - communication_edges: A list of tuples of identifiers representing the communication edges.
    - task_edges: A list of tuples of identifiers representing the task edges.

    Returns:
    - network: A networkx Graph object representing the network.
    """
    
    nodes = set()
    for edge in communication_edges + task_edges :
        nodes.add(edge[0])
        nodes.add(edge[1])
    
    task_graph          = nx.Graph()
    communication_graph = nx.Graph()
    full_network        = nx.complete_graph(nodes)
    
    
    for task_edge in task_edges:
        task_graph.add_edge(task_edge[0], task_edge[1])
        task_graph[task_edge[0]][task_edge[1]][MANAGER] = TaskManager(edge_tuple=task_edge)
    
    for comm_edge in communication_edges:
        communication_graph.add_edge(comm_edge[0], comm_edge[1])
    
    return Network(communication_graph=communication_graph, task_network=task_graph,full_network=full_network)
    


def get_network_from_states(states:dict[Identifier,np.ndarray] , communication_radius, add_task_edges_too :bool = False ) -> nx.Graph :
    """ 
    Creates a communication graph based on the states given in a dictionary.
    Note that the states are assumed to be given such that the first two elements are the x,y position of the agent.
    
    Args:
    ----
    states (dict[Identifier,np.ndarray]): 
        A dictionary where the keys are agent IDs and the values are numpy arrays representing the states of the agents.
        
    communication_radius (float): 
        The communication radius for the agents.
    add_task_edges_too (bool, optional):
        Whether to add task edges to the network too based on the communication ones. Defaults to False.
    
    """
    
    comm_graph = nx.Graph()
    comm_graph.add_nodes_from(states.keys())
    full_network = nx.complete_graph(states.keys())
    
    if add_task_edges_too:
        task_graph = nx.Graph()
        task_graph.add_nodes_from(states.keys())
    
    
    others = states.copy()
    for id_i,state_i in states.items() :
        others.pop(id_i)
        for id_j,state_j in others.items():
            distance = np.linalg.norm(state_i[:2]-state_j[:2])
            if distance < communication_radius:
                comm_graph.add_edge(id_i,id_j)  
                if add_task_edges_too:
                    task_graph.add_edge(id_i,id_j)
                    task_graph[id_i][id_j][MANAGER] = TaskManager(edge_tuple=(id_i,id_j)) 
    
    return Network(communication_graph=comm_graph, task_network=task_graph, full_network=full_network)


    
#!to be revised
# def add_communication_maintenance_tasks(task_graph:nx.Graph, communication_radius:float, agents: dict[Identifier,DynamicalModel|Agent],interval_of_enforced_communication:TimeInterval = TimeInterval(a=0,b=10**6)) :
#     """
#     Adds communication maintenance tasks to the task graph.

#     Args :
#     - task_graph (nx.Graph): The task graph to which the communication maintenance tasks will be added.
#     - communication_radius (float): The communication radius for the agents.
#     - agents (dict[Identifier,DynamicalModel|Agent]): A dictionary of agents, where the keys are agent IDs and the values are either DynamicalModel objects or Agent objects.
#     - init_time (float, optional): The initial time for the tasks. Defaults to 0.

#     Returns:
#     - task_graph (nx.Graph): The updated task graph with the communication maintenance tasks added.
#     """
    
#     if all([isinstance(obj,Agent) for obj in agents.values()]): # just extract the model
#         agents_models = {agent.unique_identifier:agent.dynamical_model for agent in agents.values()}
#     else:
#         agents_models = agents
    
    
#     for i,j,att in task_graph.edges(data=True) :
#         # create a predicate for each edge
#         source  = i
#         target  = j
#         edgeObj = att["container"] # contains the task for the given edge
        
#         if target != source:
#             agent_model_i = agents_models[source]
#             agent_model_j = agents_models[target]
#             # create communication maintenance predicate
#             predicate = epsilon_position_closeness_predicate(model_agent_i = agent_model_i,
#                                                             model_agent_j = agent_model_j, 
#                                                             epsilon       = communication_radius)
            
#             # set G_[0,infty] as the time interval
#             temporal_op = AlwaysOperator(time_interval=interval_of_enforced_communication)
#             # create a task for each edge
#             task      = StlTask(predicate = predicate, temporal_operator = temporal_op,name = f"Communication_{edgeObj.edge_tuple}")
#             # add the task to the edge
#             edgeObj.add_tasks(task)
    
#     return task_graph

