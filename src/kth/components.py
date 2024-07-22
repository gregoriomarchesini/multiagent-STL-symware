import numpy as np
import casadi as ca
import casadi.tools as ca_tools
from   typing import Iterable,Callable
from   enum import Enum
from dataclasses import dataclass
import asyncio
import time

from kth.signal_temporal_logic import *
from kth.dynamics              import InputAffineDynamicalSymbolicModel,SingleIntegrator,DifferentialDrive
from kth.optimization          import ImpactSolverLP
from kth.utils                 import NoStdStreams,PIDdrone
from kth.network               import Network


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

try:
    from symaware.simulators.pybullet import (
        Environment,
        DroneRacerEntity,
        DroneCf2pEntity,
        DroneCf2xEntity,
        DroneRacerModel,
        DroneCf2xModel,
        DroneModel,
    )
except ImportError as e:
    raise ImportError(
        "symaware-pybullet non found. "
        "Try running `pip install symaware-pybullet` or `pip install symaware[simulators]`"
    ) from e




class LeadershipToken(Enum) :
    """Enumeration class for the three types of edge leadership tokens"""
    LEADER    = 1
    FOLLOWER  = 2
    UNDEFINED = 0
    


@dataclass(frozen=True)
class StateMessage(Message):
    """A message that contains the state of an agent"""
    state: np.ndarray
    time_stamp: float = 0.0


# dictionary
class STLKnowledgeDatabase(KnowledgeDatabase):
    stl_tasks          : list[StlTask]              # stores the task for an agent in a list
    leadership_tokens  : dict[int,LeadershipToken]                       # stores the leadership tokens for the agent
    
    
@dataclass(frozen=True)
class StateMessage(Message):
    """A message that contains the state of an agent"""
    state: np.ndarray
    time_stamp: float = 0.0
    
    

class ControlMessageType(Enum) :
    BEST_IMPACT            = 2
    WORSE_IMPACT           = 3
    MAX_EXPRESSED_VELOCITY = 4
    
    
@dataclass(frozen=True)
class ControlMessage(Message):
    """A message that contains the state of an agent"""
    type      : int
    time_stamp: float = 0.0
    value     : float = 0.0


class MyPerceptionSystem(PerceptionSystem):
    
    def __init__(self, agent_id: int, environment: Environment, async_loop_lock: "Tasynclooplock | None" = None):
        
        super().__init__(agent_id,environment, async_loop_lock)
        self.current_agent_velocity = np.zeros(2)
        
    def _compute(self) -> dict[Identifier, StateObservation]:
        
        # update velocity of 
        self.current_agent_velocity = self._env.get_agent_state(self._agent_id)[7:9] # state of agent is [x,y,z,q1,q2,q3,q4,vx,vy,vz,wx,wy,wz]. So this takes vx,vy
        return   {agent_id : state[:2] for agent_id, state in  self._env.agent_states} # we only need the x-y position of the agents

    

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
            
            controller = self._agent.controllers[0] # get reference to the controller of the agent
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
        controller = self._agent.controllers[0] # get reference to the controller of the agent
        
        if message.type == ControlMessageType.BEST_IMPACT :
            controller._best_impact_from_leaders[message.sender_id] = message.value
        elif message.type == ControlMessageType.WORSE_IMPACT :
            controller._worse_impact_from_follower = message.value
       
            
        
    async def async_stop(self):
        self._get_message_task.cancel()
        return await super().async_stop()
    
        

class HighLevelController(Controller):
    """Implementation of Leader Based Controller for satisfaction of STL tasks in Multi-Agent Systems"""
    
    def __init__(self, identifier                 : Identifier,
                       mathematical_model         : InputAffineDynamicalSymbolicModel,
                       enable_collision_avoidance : bool = False,
                       collision_sensing_radius   : float = float("inf"),
                       async_loop_lock: "Tasynclooplock | None" = None) -> None:
        """
        Initialize controller

        Args:
            identifier (Identifier): single agent identifier
            mathematical_model (InputAffineDynamicalSymbolicModel): mathematical model of the agent to compute the optimal control input
            enable_collision_avoidance (bool, optional): Flag that defines if the agent will implement a collision avoidance strategy or not. Defaults to False.
            collision_sensing_radius (float, optional): Defines the radius that the agent uses to decide of another agent/object is in possible collision. Defaults to float("inf").
            async_loop_lock (Tasynclooplock | None, optional): loop lock for the component. Defaults to None. #! todo: Ask Ernesto if we should use a continuous loop lock or if we should use a trigger here.

        Raises:
            ValueError: if the mathematical model identifier is different from the agent identifier
        """
        
        # initialise controller wil dynamica models and agent identifier
        super().__init__(identifier, async_loop_lock)
        
        
        if mathematical_model.ID != identifier:
            raise ValueError(f"the mathematical model identifier is {mathematical_model.ID} but the agent identifier is {identifier}. They should be the same")
        
        self._sensing_radius = collision_sensing_radius
        self._mathematical_model       : InputAffineDynamicalSymbolicModel    = mathematical_model # mathematical model applied to find the optimal control input
        self._stl_tasks                : list[StlTask]               = []    # list of all the tasks to be satisfied
        self._leadership_tokens        : dict[int,LeadershipToken]   = {}    # leadership tokens [neighbour_id, token]
        
        # information stored about your neighbours
        self._task_neighbours_id         : set[int] = set()    # neighbouring agents IDS in the task graph
        
        # Impact information
        self._best_impact_from_leaders   : dict[int,float] = {}  # for each leader you will receive a best impact that you can use to compute your gamma
        self._worse_impact_on_leaders    : dict[int,float] = {}  # after you compute gamma you send back your worse impact to the leader
        self._worse_impact_from_follower : float = None # this is the worse impact that the follower of your task will send you back
        self._best_impact_on_follower    : float = None # this is the best impact that you can have on the task you are leading


        # Barrier functions
        self._input_constraints           : list[ca.MX]                  = []      # Input constraints
        self._barrier_constraints         : list[ca.MX]                  = []      # barrier constraint
        self._barrier_functions           : list[BarrierFunction]        = []      # for each task neighbour you have a barrier function to satisfy
        self._barrier_you_are_leading     : BarrierFunction              = None    # this is the barrier that is used to make sure that the agent is leading
        self._barriers_you_are_following  : list[BarrierFunction]        = []      # this is the list of barriers that the agent is following
        self._independent_barrier         : BarrierFunction              = None  # this is the barrier that defines self tasks
        
        # collisions alpha_factor 
        self._collision_alpha_factor = 0.5 # this is the alpha factor used to define the barrier functions for collision avoidance
        
        self._slack_vars              : dict[int,ca.MX] = {}            # list of slack variables for each edge [i,j]. the key is [j]
        self._warm_start_sol          : ca.DM|np.ndarray = np.array([]) # warm start solution for the optimization problem
        self._initialization_time     : float  = None                   # time of initialization for the simulation
        
        # For computation of control reduction factor
        self._impact_solver           : ImpactSolverLP   = ImpactSolverLP(model=self._mathematical_model) # this is a minimizer function that will be used to compute best and worse impact 
        self._gamma                    :float            = 1      # 0 < gamma_i <= 1 control reduction function
        self._gamma_tilde              :dict[int,float]  = {}     # list of all the intermediate gammas for each task. This corresponds to gamma_tilde_ij. It will be default to int:None at every iteration
        self._follower_neighbour        :int              = None   # this is the identifier of the only agent that is your follower! (if you have one)
        self._leader_neighbours         :list[int]        = []     # this is the list of leader neighbours for the tasks around you
        
        self._task_neighbour_agents_models : dict[int,InputAffineDynamicalSymbolicModel] = {}
        
        # check if tasks are given to this controller
        self._has_tasks                 = False
        self._has_self_tasks            = False
        self._gamma_is_available        = False
        
        # collision checks
        self._enable_collision_avoidance = enable_collision_avoidance
        self._primary_parameters         : ca_tools.structure3.msymStruct = None # these are the parameters of the primary controller
        self._backup_parameters              : ca_tools.structure3.msymStruct = None # these are the parameters of the back up controller that only takes care of the collision avoidance
        self._primary_solver             : ca.Function = None # this is the primary controller solver used to compute the control input for tasks and collision avoidance
        self._backup_solver                  : ca.Function = None # this is the backup controller solver used to compute the control input for collision avoidance only
    
        self.__LOGGER = get_logger(__name__, f"Controller {self._agent_id}")
        
    #exposed attributes according to main abstract class
    @property
    def mathematical_model(self) -> InputAffineDynamicalSymbolicModel:
        return self._mathematical_model
    @property
    def barrier_functions(self) -> dict[int,BarrierFunction]:
        """barrier functions"""
        return self._barrier_functions
    @property 
    def gamma(self) -> float:
        return self._gamma
    @property 
    def follower_neighbour(self):
        return self._follower_neighbour
    @property
    def leader_neighbours(self):
        return self._leader_neighbours
    @property
    def follower_neighbour(self):
        return self._follower_neighbour
    @property
    def leader_neighbours(self):
        return self._leader_neighbours
    @property
    def is_ready_to_compute_gamma_and_best_impact(self)-> bool :
        return all([self._gamma_tilde[identifier] is not None for identifier in self.leader_neighbours])
    @property
    def task_neighbours_id(self) -> list[int]:
        return self._task_neighbours_id
    
    @property
    def is_ready_to_compute_optimal_input(self) -> bool:
        return self._worse_impact_from_follower is not None
    
    def initialise_component(
        self,
        agent: "Agent",
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase,
    ):  
        
        
        self._dynamical_model = agent.model
        self.__LOGGER.info("Setting up the controller for agent %d", self._agent_id)
         
        try :
            self.add_tasks(initial_knowledge_database[self.agent_id]["stl_tasks"]) # save the tasks
        except KeyError as e:
            self.__LOGGER.error("The knowledge database does not contain the STL tasks. Please check the knowledge database and make sure it contains the key ""stl_tasks""")
            raise e
        try : # save the leadership tokens for each agent
            self._leadership_tokens = initial_knowledge_database[self.agent_id]["leadership_tokens"] # save the leadership tokens
        except KeyError as e:
            self.__LOGGER.error("the knowledge database does not contain the leadership tokens. Please check the knowledge database and make sure it contains the key ""leadership_tokens""")
            raise e
        
        count = 1
        for identifier,token in self._leadership_tokens.items() :
            if token is LeadershipToken.UNDEFINED :
                raise RuntimeError(f"The leadership token for agent {self._agent_id} is undefined. Please check that the token passing algorithm gives the correct result. No undefined tokens can be present")
            elif token is LeadershipToken.LEADER :
                self._follower_neighbour = identifier # get the follower neighbour
                if count > 1:
                    raise RuntimeError(f"Agent {self._agent_id} has more than one follower neighbour. This is not allowed. Please check the token passing algorithm")
                count += 1
            elif token is LeadershipToken.FOLLOWER :
                self._leader_neighbours += [identifier] # get the leader neighbours
        
        self._best_impact_from_leaders = {leader:None for leader in self._leader_neighbours} # set the best impact to None for all the leader neighbours
        self._gamma_tilde              = {leader:None for leader in self._leader_neighbours} # set the gamma tilde to None for all the leader neighbours
        
        # we can now define the task neighbours 
        for task in self._stl_tasks :
            
            if len(task.contributing_agents) == 1 : # you are aware that you have self tasks
                self._has_self_tasks = True
                
            elif len(task.contributing_agents) >1 : # you are not a task neighbour of yourself
                i,j = task.contributing_agents # then there are two agents instead
                
                if i == self._agent_id:
                    self._task_neighbours_id.add(j)
                else:
                    self._task_neighbours_id.add(i)

        self._initial_awareness_database  = initial_awareness_database # save the awareness database
        communication_system : Transmitter = agent.communication_sender
        try :
            network = communication_system.network
        except :
            raise RuntimeError("Communication systems does not have an Accociated Netwoork. Please check the communication system")
        self._max_num_possible_collision = len(network.full_network.nodes)  # tells how many you are prepared to deal with. If you have more and error will be raised
        self._initialization_time = 0.                                      # time at which the problem is initialized
        self._collision_constraint_fun :ca. Function =  self._get_collision_avoidance_barrier()  # used to evaluate the collision avoidance constraint
        
        # Get primary controller solver
        self._primary_parameters   = self.get_primary_controller_parameters()
        self._primary_solver  = self.get_primary_controller_solver()
        
        # Get backup controller solver. (If collision avoidance is enabled)
        if self._enable_collision_avoidance :
            self._backup_parameters = self.get_backup_controller_parameters()
            self._backup_solver = self.get_backup_controller_solver()
        else :
            self._backup_solver = None
            
        
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database) # call the parent class method to notify the subscribers
        
            
    def _add_single_task(self, task: StlTask) -> None:  
        """
        Adds tasks to the task list to be satisfied by the controller.
        
        Args:
            task (StlTask): Stl task to be satisfied
        Raises: 
            ValueError: if the task does not involve the state of the current agent
        """
        if isinstance(task, StlTask):
            if  not (self._agent_id in task.contributing_agents ) :
                raise ValueError(f"Tasks do not involve the state of the current agent. Agent index is {self._agent_id}, but task is defined over agents ({task.contributing_agents})")
            else:
                contributing_agents = task.contributing_agents
                if len(contributing_agents) > 1:
                    self.__LOGGER.info(f"Added tasks over edge : {task.contributing_agents}")
                    self._stl_tasks.append(task)
                else:
                    self.__LOGGER.info(f"Added self task : {[task.contributing_agents[0],task.contributing_agents[0]]}")
                    self._stl_tasks.append(task)
                   
        else:
            raise Exception(f"Valid type is {StlTask.__name__}")
        
        
    def add_tasks(self, tasks: list[StlTask] | StlTask) -> None:
        """Adds tasks to the task list to be satisfied by the controller."""
        if isinstance(tasks, list):
            for task in tasks:
                self._add_single_task(task=task)
        self._has_tasks = True
        
        
    
    def get_primary_controller_solver(self) -> ca.Function:
        """ 
        Constructs the primary optimization solver. 
        This solver contains both the barrier functions constraints and the collision avoidance constraints if these are enabled.
        """
        
        constraints = []
        A = self._mathematical_model.input_constraints_A
        b = self._mathematical_model.input_constraints_b
        
        input_constraints         : ca.MX  = A@self._mathematical_model.control_input_sym - self._primary_parameters["gamma"]*b  # Ax-b <=0
        self._barrier_constraints : ca.MX  = self._get_barrier_constraints()
        
        constraints += [self._barrier_constraints]                  
        constraints += [self._get_slack_positivity_constraints()]   
        constraints += [input_constraints]
        
        if self._enable_collision_avoidance :
            constraints += [self._get_collision_avoidance_constraints(self._primary_parameters)]
        
        # create vectors of constraints and optimization variables
        constraints_vector = ca.vertcat(*constraints)
        slack_vector       = ca.vertcat(*list(self._slack_vars.values()))
        opt_vector         = ca.vertcat(self._mathematical_model.control_input_sym,slack_vector)
        
        cost = self._mathematical_model.control_input_sym.T @  self._mathematical_model.control_input_sym # classic quadratic cost
        
        for identifier,slack in self._slack_vars.items():
            if identifier == self._agent_id:
                cost += 100* slack**2  
            else :
                cost += 10* slack**2
       
    
        # report information 
        info = "-\n-------------------------------------------------------------------------------------------------\n"
        info += "Primary Controller Optimization Program Built Info : \n"
        info += "--------------------------------------------------------------------------------------------------\n"
        info += f"Number of variables                 : {opt_vector.shape[0]} \n"
        info += f"Number of control variables         : {self._mathematical_model.control_input_sym.shape[0]} \n"
        info += f"Number of slack variables           : {slack_vector.shape[0]} \n"
        info += f"Number of constraints               : {constraints_vector.shape[0]} \n"
        info += f"Number of barrier constraints       : {self._barrier_constraints.shape[0]} \n"
        
        self.__LOGGER.info(info)
        
        # creates a quadratic solver for nominal controller
        try :
            with NoStdStreams() :
                qp = {'x':opt_vector, 
                      'f':cost, 
                      'g':constraints_vector ,
                      'p':self._primary_parameters}
                
                primary_solver = ca.qpsol('S', 'qpoases', qp,{"printLevel":"none"}) # create a solver object
                
        except Exception as e:
            self.__LOGGER.error(f"Error in creating the solver object. Following Error was reported {str(e)}")
            raise e
    
        return primary_solver
    
    def get_backup_controller_solver(self) :
        """ 
        Constructs the backup optimization solver. 
        This solver only contains the collision avoidance solver and it is used when the primary solver does not fund a solution. 
        Hence, in the case the primary solver does not find a solution, the backup solver is called, which neglects all the task and only takes care of avoiding all the 
        other agents or objects in the environment.
        """
        
    
        A = self._mathematical_model.input_constraints_A
        b = self._mathematical_model.input_constraints_b
        
        try :
            input_constraints = A@self._mathematical_model.control_input_sym - b  # Ax-b <=0
        except Exception as e:
            self.__LOGGER.error(f"There is a probable error in the input constraint matrices. Check the size of the matrices `input_constraints_A` and `input_constraints_b` for agent {self._agent_id}.")
            raise e
        collision_constraints = self._get_collision_avoidance_constraints(self._backup_parameters)
        
        
        # We now built the optimization program
        constraints  = ca.vertcat(input_constraints,collision_constraints)      # stacking constraints, variables and bounds
        opt_vector   = self._mathematical_model.control_input_sym
        
        if isinstance(self._mathematical_model,SingleIntegrator):
            cost = self._mathematical_model.control_input_sym.T @  self._mathematical_model.control_input_sym # classic quadratic cost
        elif isinstance(self._mathematical_model,DifferentialDrive):
            H = np.array([[1000/self._mathematical_model.max_speed**2,0],[0, 1/self._mathematical_model.max_angular_velocity**2]]) # penalise use of longitudianal speed 
            cost = self._mathematical_model.control_input_sym.T @H @ self._mathematical_model.control_input_sym
        
        
        # report information 
        info = "-\n-------------------------------------------------------------------------------------------------\n"
        info += "Backup Controller Optimization Program Built Info : \n"
        info += "--------------------------------------------------------------------------------------------------\n"
        info += f"Number of control variables         : {opt_vector.shape[0]} \n"
        info += f"Number of control input constraints : {input_constraints.shape[0]} \n" # this is a vector constraint inisde a list
        info += f"Number of collision constraints     : {collision_constraints.shape[0]} \n" # this is a vector constraint inisde a list
        info += "Note : This controller will only take care of collision avoidance. No task is included here. No constraint can be satisfied as a slack variable"
        
        self.__LOGGER.info(info)
        
        # creates a quadratic solver for nominal controller
        try :
            with NoStdStreams() :
                qp = {'x':opt_vector, 
                      'f':cost, 
                      'g':constraints ,
                      'p':self._backup_parameters}
                
                backup_solver = ca.qpsol('S', 'qpoases', qp,{"printLevel":"none"}) # create a solver object
                
        except Exception as e:
            self.__LOGGER.error(f"Error in creating the solver object for backup controller. Following Error was reported {str(e)}")
            raise e
        
        return backup_solver
        
        
    def _get_slack_positivity_constraints(self) -> ca.MX:
        """Get the constraints related to the slack variables being positive"""
        self.__LOGGER.info(f"Computing Slack Constraints {self._agent_id}")
        return - ca.vertcat(*list(self._slack_vars.values())) #slack >=0 ---> -slack <= 0
        
    def _get_barrier_constraints(self) -> ca.MX:
        """Get the barrier constraints associated with each task"""
        
        if not self._has_tasks : # no tasks to be fulfilled
            self._barrier_constraints = []
            self.__LOGGER.info(f"No tasks found. The controller will not do anything")
            return
        
        # remove outdated tasks
        self._stl_tasks = [task for task in self._stl_tasks if not(task.temporal_operator.time_of_remotion < self._initialization_time)]
        self.__LOGGER.info(f"Starting transformation from Task to Barrier Function for agent {self._agent_id}")
        
        self._barrier_functions : list[BarrierFunction] = self.generate_barrier_from_tasks( tasks = self._stl_tasks) # create the barrier functions
        barrier_constraints     : ca.MX                 = self.generate_barrier_constraints(barriers = self._barrier_functions) # create the barrier constraints
        
        self._barrier_you_are_leading,self._barriers_you_are_following ,self._independent_barrier  = self._get_splitted_barriers() 
        self.__LOGGER.info(f"Following {len(self._barriers_you_are_following)} barriers")
        
        if self._barrier_you_are_leading != None:
            self.__LOGGER.info(f"Leading one barrier")
        if self._independent_barrier != None:
            self.__LOGGER.info(f"One independent task detected")
            
        return barrier_constraints
    
    
    
    def _get_collision_avoidance_barrier(self)->None :
        
        x  = ca.MX.sym("x",self._mathematical_model.state_vector_sym.shape[0]) # state of the agent (which also constains the position)
        y  = ca.MX.sym("y",2) # position of the obstacke
        switch = ca.MX.sym("switch",1) # switch off the constraint when not needed
        load   = ca.MX.sym("load",1)
        
        collision_radius = 0.5 # assuming the two agents are 1m big
        
        barrier = (x[:2]-y).T@(x[:2]-y) - (2*collision_radius)**2 # here the collsion radius is assumed to be 1 for each object 
        f_x = self._mathematical_model.f_fun(x)
        g_xu = self._mathematical_model.g_fun(x)@self._mathematical_model.control_input_sym
        
        db_dx = ca.jacobian(barrier,x)
        
        constraint =  db_dx@(f_x + g_xu) +  load*(  self._collision_alpha_factor * barrier) # if load = 0.5 -> cooperative collsion. If loead =1 , then non cooperative
        
        #(-1) factor needed to turn the constraint into negative g(x)<=0
        # switch -> it will be 1 if it is needed the constraint and 0 if not
        collision_constraint_fun = ca.Function("collision_avoidance",[x,y,switch,load,self._mathematical_model.control_input_sym],
                                          [-1*(constraint)*switch]) # create a function that will be used to compute the collision avoidance constraints
        
        return collision_constraint_fun
            
    
    def _get_collision_avoidance_constraints(self,parameters) -> ca.MX:
        """ Here we create the collision avoidance solver """
        
        
        collision_contraints = []
        for jj in range(self._max_num_possible_collision):
            collision_contraints += [self._collision_constraint_fun( parameters["state_"+str(self._agent_id)],
                                                                     parameters["collision_pos_"+str(jj)],
                                                                     parameters["collision_switch_"+str(jj)],
                                                                     parameters["collision_load_"+str(jj)],
                                                                     self._mathematical_model.control_input_sym)]
            
        
        return ca.vertcat(*collision_contraints)
    
    
    
    def get_primary_controller_parameters(self) -> ca_tools.structure3.msymStruct:
        """This function returns the parameters of the primary controller. The primary controller is the one that takes care of the main tasks and the collision avoidance (in case this is activated).
        The parameters are the following :
        - one parameters for the self state of the agent
        - one parameter for each task neighbour state
        - one gamma parameter for the control reduction
        - time parameter
        - one state parameter for each obstacle met by the agent (these could be other agents or static obstacles)
        - one switch parameter for each obstacle you might meet. This is useful to switch off the collision avoidance constraint when not needed (you have up to self._max_num_possible_collision obstacles to meet)
        - one epsilon factor (worse impact on barrier) for the follower agent (if you have one)
        """
        parameters_list = []
        parameters_list +=  [ca_tools.entry('state_'+str(identifier),shape=self._task_neighbour_agents_models[identifier].state_vector_sym.shape[0]) for identifier in self._task_neighbours_id] # one parameter for each task neighbour state
        parameters_list +=  [ca_tools.entry('state_'+str(self._agent_id),shape=self._mathematical_model.state_vector_sym.shape[0])]                                        # one parameter for the self state
        parameters_list +=  [ca_tools.entry('gamma',shape=1)] # one gamma parameter for the control reduction
        parameters_list +=  [ca_tools.entry('time',shape=1)] # time parameter
        
        
        if self._enable_collision_avoidance :
            parameters_list +=  [ca_tools.entry('collision_pos_'+str(identifier),shape=2) for identifier in range(self._max_num_possible_collision)]  # one parameter for the state of any obstacle met by the agent
            parameters_list +=  [ca_tools.entry('collision_switch_'+str(identifier),shape=1)  for identifier in range(self._max_num_possible_collision)]  # used to switch off a collision avoidance constraint when not needed
            parameters_list +=  [ca_tools.entry('collision_load_'+str(identifier),shape=1)  for identifier in range(self._max_num_possible_collision)]  # used to switch off a collision avoidance constraint when not needed
            
    
        if self._follower_neighbour != None : # meaning that there is one task you are leader and then you need to add the epsilon parameter (worse impact from the follower agent) in the optimization program
            parameters_list += [ca_tools.entry('epsilon',shape=1)] 
                
        return ca_tools.struct_symMX(parameters_list) 
    
    def get_backup_controller_parameters(self) -> ca_tools.structure3.msymStruct:
        """
        This function returns the parameters for the secondary controller ).
        The parameters are the following :
        - one parameters for the self state of the agent
        - time parameter
        - one state parameter for each obstacle met by the agent (these could be other agents or static obstacles)
        - one switch parameter for each obstacle you might meet. This is useful to switch off the collision avoidance constraint when not needed (you have up to self._max_num_possible_collision obstacles to meet)
        
        note : since the dynamics are time invariant and the barrier for collision avoidance is time invariant, the you don't need to pass time to the secondary controller
        """
        
        parameters_list = []
        parameters_list +=  [ca_tools.entry('state_'+str(self._agent_id),shape=self._mathematical_model.state_vector_sym.shape[0])]                                        # one parameter for the self state
        parameters_list +=  [ca_tools.entry('collision_pos_'+str(identifier),shape=2) for identifier in range(self._max_num_possible_collision)]  # one parameter for the state of any obstacle met by the agent
        parameters_list +=  [ca_tools.entry('collision_switch_'+str(identifier),shape=1)  for identifier in range(self._max_num_possible_collision)]  # used to switch off a collision avoidance constraint when not needed
        parameters_list +=  [ca_tools.entry('collision_load_'+str(identifier),shape=1)  for identifier in range(self._max_num_possible_collision)]  # used to switch off a collision avoidance constraint when not needed
        
        return ca_tools.struct_symMX(parameters_list) 
    
    def generate_barrier_from_tasks(self,tasks: list[StlTask]) -> list[BarrierFunction]:  
        """ Given a list of StlTasks, this function generates the barrier functions associated with each task.
        Args:
            tasks (list[StlTask]): list of tasks to be satisfied by the controller
        Returns:
            barriers_list (list[BarrierFunction]): list of barrier functions associated with each
        """
        
        barriers_list = []
        # we use the same alpha function for all the barriers. This can be changed
        
        dummy_scalar    = ca.MX.sym("x",1) # dummy scalar variable to be used in the definition of the alpha function
        scale_factor    = 2
        alpha_fun = ca.Function('alpha_fun',[dummy_scalar],[scale_factor*dummy_scalar]) # this is the alpha function used to define the barrier functions
        
        for task in tasks:
            involved_agents = task.contributing_agents
            initial_conditions = {}
            
            # generate initial conditions
            for identifier in involved_agents:
                initial_conditions[identifier] = self._initial_awareness_database[identifier].state # takes the x-y position only
            
            barriers_list += [create_barrier_from_task(task               = task,
                                                       initial_conditions =  initial_conditions, 
                                                       alpha_function     = alpha_fun,
                                                       t_init             = self._initialization_time )]
             
        
        self.__LOGGER.info(f"All task successfully converted to barrier constraints")
        
        # now we reduce the number of barriers by joining the ones that are along the same edge 
        before_conjunction = len(barriers_list)
        barriers_list = self._join_conjunctions_along_same_edge(barriers_list)
        after_conjunction = len(barriers_list)
        
        if before_conjunction != after_conjunction:
            self.__LOGGER.info(f"Conjunctions of barriers along the same edge successfully created. {before_conjunction} barriers were joined into {after_conjunction} barriers")
        
        
        return barriers_list
    
    
    def generate_barrier_constraints(self,barriers:list[BarrierFunction]) -> ca.MX:
        """
        Given a list of BarrierFunctions, this function generates the constraints associated with each barrier function and stacks all the constraints together into a single vector.
        The constraint for each task is obtained according to the following three different procedures :
        case 1: single agent task
        case 2: collaborative task and the agent is the leader
        case 3: collaborative task and the agent is the follower
        
        #CASE 1:
        The barrier function is in the form b(x_i) hence the barrier constraint takes the form
        nabla_xi b(x_i) (f_i+g_i*u_i) + db_dt + alpha(b) >=0   => (sign changed) => -nabla_xi b(x_i) (f_i+g_i*u_i) - db_dt - alpha(b) <=0
        
        #CASE 2 :
        The barrier function is in the form b(x_i,x_j) hence the barrier constraint takes the form
        nabla_xi b(x_i,x_j) (f_i+g_i*u_i) + nabla_xj b(x_i,x_j) (f_j+g_j*u_j) + db_dt + alpha(b) >=0   => (sign changed) => -nabla_xi b(x_i,x_j) (f_i+g_i*u_i) - nabla_xj b(x_i,x_j) (f_j+g_j*u_j) - db_dt - alpha(b) <=0
        
        The current agent (i) is leader of the task so it will take care of the task all by itself using the worse impact from the follower (a.k.a min_{u_j}  nabla_xj b(x_i,x_j) (f_j+g_j*u_j)). Hence the constraints will be
        nabla_xi b(x_i,x_j) (f_i+g_i*u_i) + worse_imapct_j - db_dt - alpha(b) <=0
        
        #CASE 3 :
        The barrier function is in the form b(x_i,x_j) hence the barrier constraint takes the form 
        nabla_xi b(x_i,x_j) (f_i+g_i*u_i) + nabla_xj b(x_i,x_j) (f_j+g_j*u_j) + db_dt + alpha(b) >=0   => (sign changed) => -nabla_xi b(x_i,x_j) (f_i+g_i*u_i) - nabla_xj b(x_i,x_j) (f_j+g_j*u_j) - db_dt - alpha(b) <=0
        
        This time the agent is follower of the task so it will only satify his section of the task with the leader taking care of the rest. A slack variable is applied to relax the constraint.
        nabla_xj b(x_i,x_j) (f_j+g_j*u_j) + db_dt + alpha(b) + slack >=0
        """
        
        constraints = []
        for barrier in barriers :
            involved_agent = barrier.contributing_agents # for this controllere there can be only two agents involved. This is already checked before this point
            
            if len(involved_agent) > 1:
                if involved_agent[0] == self._agent_id:
                    neigbhour_id = involved_agent[1]
                else:
                    neigbhour_id = involved_agent[0]
            else : # in case it is aself task
                neigbhour_id = self._agent_id
            
            
            named_inputs = {"state_"+str(identifier):self._primary_parameters["state_"+str(identifier)]  for identifier in barrier.contributing_agents}
            named_inputs["time"] = self._primary_parameters["time"] # add the time  
            
            nabla_xi_fun                : ca.Function   = barrier.gradient_function_wrt_state_of_agent(self._agent_id) # this will have the the same inputs as the barrier itself
            partial_time_derivative_fun : ca.Function   = barrier.partial_time_derivative
            barrier_fun                 : ca.Function   = barrier.function
            dyn                         : ca.Function   = self._mathematical_model.dynamics_fun
            
            self._nabla_xi_fun = nabla_xi_fun

            # just evaluate to get the symbolic expression now
            nabla_xi = nabla_xi_fun.call(named_inputs)["value"] # symbolic expression of the gradient of the barrier function w.r.t to the state of the agent
            dbdt     = partial_time_derivative_fun.call(named_inputs)["value"] # symbolic expression of the partial time derivative of the barrier function
            alpha_b  = barrier.associated_alpha_function(barrier_fun.call(named_inputs)["value"]) # symbolic expression of the barrier function
            dynamics = dyn.call({"state":self._primary_parameters["state_"+str(self._agent_id)],"input":self._mathematical_model.control_input_sym})["value"] # symbolic expression of the dynamics of the agent
            switch_function = barrier.switch_function # symbolic expression of the barrier function
            
            # NOTE : the -1* is included to transform all the barrier constraints into g(x)<=0. 
            
            # CASE 3: the agent is follower.
            if neigbhour_id in self._leader_neighbours: # they will take care of the rest of the barrier
                
                slack             = ca.MX.sym(f"slack",1) # create slack variable (it will be forced to be strictly positive)
                self._slack_vars[neigbhour_id] = slack            
                load_sharing      = 0.1
                
                if switch_function != None :
                    barrier_constraint   = -1* ( ca.dot(nabla_xi.T, dynamics ) + load_sharing * (dbdt + alpha_b) + slack ) * switch_function(self._primary_parameters["time"])
                else :
                    barrier_constraint   = -1* ( ca.dot(nabla_xi.T, dynamics ) + load_sharing * (dbdt + alpha_b) + slack )
                
                constraints += [barrier_constraint] # add constraints to the list of constraints
                
            # CASE 2: the agent is leader.
            elif neigbhour_id == self._follower_neighbour: # No slack satisfaction here
                
                if switch_function != None :
                    barrier_constraint  = -1* ( ca.dot(nabla_xi.T, dynamics ) + dbdt + alpha_b +  self._primary_parameters['epsilon']) * switch_function(self._primary_parameters["time"])
                else :
                    barrier_constraint  = -1* ( ca.dot(nabla_xi.T, dynamics ) + dbdt + alpha_b +  self._primary_parameters['epsilon'])
                
                constraints += [barrier_constraint] # add constraints to the list of constraints
            
            # CASE 1: the agent is leader.
            elif  neigbhour_id == self._agent_id: # then it is an independent task
                
                slack                = ca.MX.sym(f"slack",1) # create slack variable (it will be forced to be strictly positive)
                self._slack_vars[self._agent_id]    = slack # higher cost for independent task to not be fulfilled

                if switch_function != None :
                    barrier_constraint   = -1 * ( ca.dot(nabla_xi.T, dynamics ) + (dbdt + alpha_b) + slack ) * switch_function(self._primary_parameters["time"])
                else :
                    barrier_constraint   = -1 * ( ca.dot(nabla_xi.T, dynamics ) + (dbdt + alpha_b) + slack ) 
                
                constraints += [barrier_constraint] # add constraints to the list of constraints
            
        return ca.vertcat(*constraints)
    
    
    
    #! todo : Ask Ernesto how can we approach the asynchronous computation of this function
    def request_best_impacts_from_leaders_and_compute_gamma_tilde_values(self):
        """
        This function computes the values of gamma tilde for each leader neighbour of the current agent. 
        Note that the self._gamma_tilde dictionary is emptied every time a new control input is computed so that new values of gamma tilde can be computed.
        #! Ask Ernesto: here what we should do is that each agent waits for all the vlues of gamma tilde and it can use a backup value if a given time out condition is
        #! met. This is important to avoid the case where the agent is stuck waiting for a value that will never come.
        
        """
        
        self.__LOGGER.info(f"Trying to compute gamma tilde values...")
        for barrier in self._barriers_you_are_following :
            involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
            
            if involved_agent[0] == self._agent_id:
                neighbour_id = involved_agent[1]
            else :
                neighbour_id = involved_agent[0]
            
            if neighbour_id in self._leader_neighbours:
                gamma_tilde = self._compute_gamma_for_barrier(barrier)
                self._gamma_tilde[neighbour_id] = gamma_tilde
                
        
    #! Missing current time
    def _compute_gamma_for_barrier(self,barrier: BarrierFunction) -> float :
        """
        Given a barrier function, and using the best impact received from the leader agent, the agent computes the gamma tilde associated with the neighbour agent involved in the task.
        A value of None is returned if the gamma_tilde for this barrier cannot be computed due to the fact that the leader agent did not transmit its best control input yet.
        
        Args:
            barrier (BarrierFunction): Barrier function for which the gamma tilde is to be computed
        
        """
    
        involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
        
        # get awareness database from the agent
        awareness_database = self._agent.awareness_database
        current_time       = self._initialization_time - time.time() #! We need a way to save the current time in the agents coordinator at the moment when all the agents where initialised
        
        if len(involved_agent) ==1:
            raise RuntimeError("The given barrier function is a self task. Gamma should not be computed for this task. please revise the logic of the contoller")
        else :
             
            if involved_agent[0] == self._agent_id:
                neighbour_id = involved_agent[1]
            else :
                neighbour_id = involved_agent[0]
        
        if not neighbour_id in self._leader_neighbours:
            raise RuntimeError(f"Agent {self._agent_id} is not a follower of agent {neighbour_id}. This is not allowed. Please check the logic of the controller")
        
        # now we need to compute the gamma for this special case
        if self._best_impact_from_leaders[neighbour_id] != None:
            neighbour_best_impact          = self._best_impact_from_leaders[neighbour_id] # this is a scalar value representing the best impact of the neighbour on the barrier given its intupt limitations
            self.__LOGGER.info(f"Unpack leader best impact from agent {neighbour_id} with value {neighbour_best_impact}") 
        else:
            self.__LOGGER.info(f"Required leaders best impact from agent {neighbour_id} not available yet. Retry later...") 
            return None
        
        
        barrier_fun    : ca.Function   = barrier.function
        local_gradient : ca.Function   = barrier.gradient_function_wrt_state_of_agent(identifier=self._agent_id)    
        
        associated_alpha_function :ca.Function  = barrier.associated_alpha_function # this is the alpha function associated to the barrier function in the barrier constraint
        partial_time_derivative:ca.Function     = barrier.partial_time_derivative
        
        if associated_alpha_function == None:
            raise RuntimeError("The alpha function associated to the barrier function is null. please remember to store this function in the barrier function object for barrier computation")

        neighbour_state       = awareness_database[neighbour_id].state    # the neighbour state
        current_agent_state  = awareness_database[self._agent_id].state  # your current state
        time                 = current_time
        named_inputs         = {"state_"+str(self._agent_id):current_agent_state.flatten(),"state_"+str(neighbour_id):neighbour_state.flatten(),"time":time}

        local_gradient_value = local_gradient.call(named_inputs)["value"].full() # compute the local gradient
  
        g_fun  : ca.Function = self._mathematical_model.g_fun
        f_fun   :ca.Function = self._mathematical_model.f_fun
        g_value = g_fun(current_agent_state).full()
        f_value = f_fun(current_agent_state).full()
        
        if f_value.shape[0] == 1:
                f_value = f_value.T
        if local_gradient_value.shape[0] == 1:
            local_gradient_value = local_gradient_value.T

    
        # this can be parallelised for each of the barrier you are a follower of
        worse_input = self._impact_solver.minimize(Lg = local_gradient_value.T @ g_value) # find the input that minimises the dot product with Lg given the bound on the input
        if worse_input.shape[0] == 1:
            worse_input = worse_input.T

        alpha_barrier_value            = associated_alpha_function(barrier_fun.call(named_inputs)["value"])                # compute the alpha function associated to the barrier function
        partial_time_derivative_value  = partial_time_derivative.call(named_inputs)["value"]                               # compute the partial time derivative of the barrier function
        zeta                           = alpha_barrier_value + partial_time_derivative_value 

        if np.linalg.norm(local_gradient_value) <= 10**-6 :
            gamma_tilde = 1
        else :
            if self._enable_collision_avoidance :
                reduction_factor = 0.8 # when you have collision avoidance you should assume that the leader might not be able to actually express its best input because he is busy avoiding someone else. So use a reduction factor
            else :
                reduction_factor = 1
                
            gamma_tilde =  -(neighbour_best_impact*reduction_factor +zeta + local_gradient_value.T @ (f_value )) / ( local_gradient_value.T @ g_value @ worse_input) # compute the gamma value
            
            if alpha_barrier_value < 0 :
                self.__LOGGER.warning(f"Alpha barrier value is negative. This entails task dissatisfaction. The value is {alpha_barrier_value}. Please very that the task is feasible")
            self.__LOGGER.info(f"gamma_{[self._agent_id,neighbour_id]} computation summary :") 
            self.__LOGGER.info(f"----------------------------------------------------------")
            self.__LOGGER.info(f"Value of the gamma_tilde :{gamma_tilde}")
            self.__LOGGER.info(f"Neighbour best impact : {neighbour_best_impact}") 
            self.__LOGGER.info(f"Alpha barrier value : {alpha_barrier_value}")
            self.__LOGGER.info(f"Partial time derivative value : {partial_time_derivative_value}")
            self.__LOGGER.info(f"Zeta value : {zeta}")
            self.__LOGGER.info(f"Local gradient value : {local_gradient_value}")
            self.__LOGGER.info(f"Value of worse impact: { local_gradient_value.T @ (f_value ) + local_gradient_value.T @ g_value @ worse_input}")
            
            
            return float(gamma_tilde)
   
    
    def compute_gamma_and_notify_best_impact_on_leader_task(self) :
        """
        Once all the gamma_tilde values are available for each leader agent, then the value of gamma is computed as the minimum of all the gamma-tilde and the the best impact on the task the agent is leader of can finally be computed and transmitted to
        the follower.
        
        Raises:
            RuntimeError: If not all the gamma_tilde values are available for each leader agent (i.e. the list of gamma_tilde values is not as long as the list of leader neighbours)
            RuntimeError: If the computed gamma value is negative. This should not happen and it breaks the process. #!todo we can find a solution to relax this as in real simulation it will probably be a problem
        """
        
        # get awareness database from the agent
        awareness_database = self._agent.awareness_database
        current_time       = self._agent.current_time #! We need a way to save the current time in the agents coordinator at the moment when all the agents where initialised
        
        # now it is the time to check if you have all the available information
        if len(self._leader_neighbours) == 0 : # leaf node case
            self._gamma = 1
        elif any([self._gamma_tilde[leader] == None for leader in self._leader_neighbours]) :
            raise RuntimeError(f"The list of gamma tilde values is not complete. some values of gamma_tilde were not computed yet")
        
        else :
            
            gamma_tildes_list = list(self._gamma_tilde.values())
            self._gamma = min(gamma_tildes_list + [1]) # take the minimum of the gamma tilde values
            
            if self._gamma<=0 :
                self.__LOGGER.error(f"The computed gamma value is negative. This breakes the process. The gamma value is {self._gamma}")
                raise RuntimeError(f"The computed gamma value for agent {self._agent_id} is negative. This breakes the process. The gamma value is {self._gamma}")
            
        
        # now that computation of the best impact is undertaken
        if self._follower_neighbour != None : # if you have a task you are leader of then you should compute your best impact for the follower agent
            self.__LOGGER.info(f"Sending Best impact notification to the follower agent {self._follower_neighbour}")
            # now compute your best input for the follower agent
            local_gradient :ca.Function = self._barrier_you_are_leading.gradient_function_wrt_state_of_agent(identifier=self._agent_id)    

            named_inputs   = {"state_"+str(self._agent_id)         :awareness_database[self._agent_id].state,
                              "state_"+str(self._follower_neighbour):awareness_database[self._follower_neighbour].state ,
                              "time"                                 :current_time}

            local_gardient_value = local_gradient.call(named_inputs)["value"].full() # compute the local gradient
            g_fun  : ca.Function = self._mathematical_model.g_fun
            f_fun   :ca.Function = self._mathematical_model.f_fun
            g_value = g_fun(awareness_database[self._agent_id].state).full()
            f_value = f_fun(awareness_database[self._agent_id].state).full()
            
            # then you are leader of this task and you need to compute your best impact
            best_input = self._impact_solver.maximize(Lg= local_gardient_value@g_value) # sign changed because you want the maximum in reality. 
            
            if f_value.shape[0] == 1:
                f_value = f_value
            if best_input.shape[0] == 1:
                best_input = best_input.T
            if local_gardient_value.shape[0] == 1:
                local_gardient_value = local_gardient_value.T

            self._best_impact_on_leader_task   = np.dot(local_gardient_value.T,(f_value + g_value @ best_input*self._gamma)) # compute the best impact of the leader on the barrier given the gamma_i
            self._best_impact_on_leader_task   = np.squeeze(self._best_impact_on_leader_task)
            
            associated_alpha_function :ca.Function  = self._barrier_you_are_leading.associated_alpha_function # this is the alpha function associated to the barrier function in the barrier constraint
            partial_time_derivative:ca.Function     = self._barrier_you_are_leading.partial_time_derivative
            
            alpha_barrier_value            = associated_alpha_function(self._barrier_you_are_leading.function.call(named_inputs)["value"])                # compute the alpha function associated to the barrier function
            partial_time_derivative_value  = partial_time_derivative.call(named_inputs)["value"]   
            zeta                           = alpha_barrier_value + partial_time_derivative_value
            
            self.__LOGGER.info(f"value of zeta : {zeta}")
            self.__LOGGER.info(f"db_dt : {alpha_barrier_value}")
            self.__LOGGER.info(f"partial time derivative : {partial_time_derivative_value}")
            
           
            # send a notification
            self.transmit_message(event="best_impact")
           
    
    def compute_and_notify_worse_impact_on_tasks_you_are_following(self) :
        """
        This function computes the the worse impact for each task that the agent is the follower of and it sends the value to the leaders. 
        The function is called once the value of gamma for the agent is computed.
        """
        
        # get awareness database from the agent
        awareness_database = self._agent.awareness_database
        current_time       = self._agent.current_time #! We need a way to save the current time in the agents coordinator at the moment when all the agents where initialised
        
        # if you have leaders to notify then do it
        if len(self._leader_neighbours) != 0 :
            self.__LOGGER.info(f"Sending worse impact notification to leaders.... ")
            
            for barrier in self._barriers_you_are_following :
                # now compute your best input for the follower agent
                involved_agent = barrier.contributing_agents # only two agents are involved in a function for this controller
                
                if len(involved_agent) > 1:
                    if involved_agent[0] == self._agent_id:
                        leader_neighbour = involved_agent[1]
                    else :
                        leader_neighbour = involved_agent[0]
                else : # single agent task doesn't need any worse_input computation
                    continue
                    
                
                local_gradient :ca.Function = barrier.gradient_function_wrt_state_of_agent(identifier=self._agent_id)    
                
                named_inputs   = {"state_"+str(self._agent_id)        :awareness_database[self._agent_id].state,
                                  "state_"+str(leader_neighbour)      :awareness_database[leader_neighbour].state,
                                  "time"                              :current_time}

                local_gardient_value = local_gradient.call(named_inputs)["value"].full() # compute the local gradient
                g_fun  : ca.Function = self._mathematical_model.g_fun
                f_fun   :ca.Function = self._mathematical_model.f_fun
                g_value = g_fun(awareness_database[self._agent_id].state).full()
                f_value = f_fun(awareness_database[self._agent_id].state).full()
                # then you are leader of this task and you need to compute your best impact
                worse_input = self._impact_solver.minimize(Lg= local_gardient_value@g_value) 
                
                
                if f_value.shape[0] == 1:
                    f_value = f_value
                if worse_input.shape[0] == 1:
                    worse_input = worse_input.T
                if local_gardient_value.shape[0] == 1:
                    local_gardient_value = local_gardient_value.T
                    
                self._worse_impact_on_leaders[leader_neighbour]  = np.dot(local_gardient_value.T,(f_value + g_value @ worse_input*self._gamma)) # compute the best impact of the leader on the barrier given the gamma_i
                
            
            # send a notification
            self.transmit_message(event="worse_impact")
    
        
    
    def transmit_message(self,event:str) ->None:
        """This function is used then by the conroller to update the information of the neighbours."""
         
        if event == "best_impact" :
            """send the value of the best impact here"""
            message = ControlMessage(sender_id = self._agent_id,receiver_id= self._follower_neighbour, time_stamp = -1, value = self._best_impact_on_follower) 
            self._agent.communication_sender.enqueue_messages(message)
            self._agent.communication_sender.async_loop_lock.trigger() # trigger message transmission
        
        elif event == "worse_impact" :
            for leader_id in self._leader_neighbours :
                message = ControlMessage(sender_id = self._agent_id,receiver_id= leader_id, time_stamp = -1, value = self._worse_impact_on_leaders[leader_id]) 
                self._agent.communication_sender.enqueue_messages(message)
            self._agent.communication_sender.async_loop_lock.trigger() # trigger message transmission
                
        else :
            raise RuntimeError(f"Event {event} is not recognized by the controller. Please check the event name")
    
    
    
    # some useful support functions 
    def _get_splitted_barriers(self) -> tuple[BarrierFunction,list[BarrierFunction],BarrierFunction]:
        """
        Takes a list of barrier functions and divides it into three output :
        - one barrier that is along the edge where the agent is a leader
        - a list of barriers that are along the edge where the agent is a follower
        - one barrier associated with the independent task
        
        Args:
            barriers (list[BarrierFunction]): A list of barrier functions
            tokens (dict[int,LeadershipToken]): A dictionary of leadership tokens
        Returns:
            barriers_you_are_leading (BarrierFunction): The barrier function that the agent is leading
            follower_barriers (list[BarrierFunction]): A list of barrier functions that the agent is following
            independent_barrier (BarrierFunction): The barrier function that is independent of the agent
        """
        barriers : list[BarrierFunction]     = self._barrier_functions,
        tokens   : dict[int,LeadershipToken] = self._leadership_tokens
        
        
        barrier_you_are_leading   = None
        follower_barriers         = []
        independent_barrier       = None
        
        for barrier in barriers:
            involved_agents = barrier.contributing_agents
            
            if len(involved_agents) == 1:
                independent_barrier = barrier
            
            if len(involved_agents) > 1:
                
                if involved_agents[0] == self._agent_id:
                    neighbour_id = involved_agents[1]
                else:
                    neighbour_id = involved_agents[0]
                
                if tokens[neighbour_id] == LeadershipToken.LEADER:
                    barrier_you_are_leading = barrier
                else:
                    follower_barriers.append(barrier)
        
        return barrier_you_are_leading,follower_barriers,independent_barrier
    
    
    def _join_conjunctions_along_same_edge(self,barriers:list[BarrierFunction]) -> list[BarrierFunction]:
        """
        Takes a list of barriers and join them in conjunctions if they are part of the same edge. Note that for this type of barriers we have 
        that only two agents are involved in the same task so we can easily check if two functions are defined over the same edge. And if they are then
        we join them into a single barrier functions and remove the two original barriers from the list
        
        Args:
            barriers (list[BarrierFunction]): A list of barrier functions
        Returns:
            new_barriers (list[BarrierFunction]): A list of barrier functions where the barriers along the same edge are joined in conjunctions
        """
        
        new_barriers = []
        
        edge_barriers         = [barrier for barrier in barriers if len(barrier.contributing_agents) > 1]
        single_agent_barriers = [barrier for barrier in barriers if len(barrier.contributing_agents) == 1]
        
        dummy_scalar    = ca.MX.sym("x",1) # dummy scalar variable to be used in the definition of the alpha function
        scale_factor    = 2
        alpha_fun = ca.Function('alpha_fun',[dummy_scalar],[scale_factor*dummy_scalar]) # this is the alpha function used to define the barrier functions
        
        # first create the conjunctions for the edge tasks
        for identifier in self._task_neighbours_id:
            barriers_along_edge = [barrier for barrier in edge_barriers if identifier in barrier.contributing_agents ]
            
            if len(barriers_along_edge) > 1 :
                new_barriers+=[conjunction_of_barriers(*barriers_along_edge,associated_alpha_function = alpha_fun)]
            else :
                new_barriers += barriers_along_edge
        
        # now the single agent tasks
        single_agent_barriers = [barrier for barrier in barriers if len(barrier.contributing_agents) == 1]
        
        if len(single_agent_barriers) > 1:
            new_barriers += [conjunction_of_barriers(*single_agent_barriers,associated_alpha_function = alpha_fun)]
        else :
            new_barriers += single_agent_barriers
            

        
        return new_barriers
    
    
    #! Fix time problem
    def compute_optimal_control_input(self)-> np.ndarray:
        
        awareness_database  = self._agent.awareness_database
        current_time        = self._agent.current_time #! We need a way to save the current time in the agents coordinator at the moment when all the agents where initialised
        knowledge_database  = self._agent.knowledge_database
         
        # -------------------------------------------  PARAMETERS FILLING ----------------------------------------------------------------------
        # we now set up all the parameters for each agent
        current_parameters_primary_controller = self._primary_parameters(0)
        if self._enable_collision_avoidance :
            current_parameters_secondary_controller = self._backup_parameters(0)
        
        for jj in self._task_neighbours_id:
            current_parameters_primary_controller["state_"+str(jj)] = awareness_database[jj].state
        
        
        current_parameters_primary_controller["time"] = current_time # fill current time
        current_parameters_primary_controller["gamma"] = self._gamma # fill the current gamma value
        current_parameters_primary_controller["state_"+str(self._agent_id)] = awareness_database[self._agent_id].state # get the current state of the agent
        
        if self._enable_collision_avoidance :
            current_parameters_secondary_controller["state_"+str(self._agent_id)] = awareness_database[self._agent_id].state # get the current state of the agent
        
        if self._follower_neighbour != None : # in case you have some follers you need to consier the worse impact among your constraints
            current_parameters_primary_controller["epsilon"] = self._worse_impact_from_follower
        
        
        # Now we add the collision avoidance parameters 
        # we will need to check which of the agents you have in your awareness database are within your sensing area
        
        if self._enable_collision_avoidance :
            pos_of_colliding_entities = {}
            self_pos = awareness_database[self._agent_id].state
            for identifier in awareness_database.keys():
                if identifier != self._agent_id:
                    
                    pos_entity = awareness_database[identifier].state
                    distance_squared = (pos_entity[0]-self_pos[0])**2 + (pos_entity[1]-self_pos[1])**2
                    
                    if distance_squared <= self._sensing_radius**2 :
                        pos_of_colliding_entities[identifier] = awareness_database[identifier].state
            
            if len(pos_of_colliding_entities) > self._max_num_possible_collision:
                raise RuntimeError(f"The number of entities in the sensing area is {len(pos_of_colliding_entities)}. This is more than the maximum number of possible collision which is {self._max_num_possible_collision}. You should revise the logic of the controller")
            # now we fill the parameters
            
            for jj,identifier in zip(range(len(pos_of_colliding_entities)),pos_of_colliding_entities.keys()):
                
                distance_squared = (pos_of_colliding_entities[identifier][0]-self_pos[0])**2 + (pos_of_colliding_entities[identifier][1]-self_pos[1])**2
                
                # fill both parameters at the same time.
                current_parameters_primary_controller["collision_pos_"+str(jj)]    = pos_of_colliding_entities[identifier]
                current_parameters_primary_controller["collision_switch_"+str(jj)] = 1
                current_parameters_primary_controller["collision_load_"+str(jj)]   = 0.5 # expect cooperation from the other agent
                
                current_parameters_secondary_controller["collision_pos_"+str(jj)]    = pos_of_colliding_entities[identifier]
                current_parameters_secondary_controller["collision_switch_"+str(jj)] = 1
                current_parameters_secondary_controller["collision_load_"+str(jj)]   = 0.5

                
                self.__LOGGER.info(f"value of jj {jj}")
            # this has just to be filled because they are inactive constraints
            
            for kk in range(jj+1,self._max_num_possible_collision) :
                
                # fill both parameters at the same time.
                current_parameters_primary_controller["collision_pos_"+str(kk)]    = np.zeros((2,1)) # here we put the value of the last agent found in the collsion radius, but it can be any nonzero random position since the constraint is not active
                current_parameters_primary_controller["collision_switch_"+str(kk)] = 0                             # Just make sure that the relatve position is not (0,0) otherwise the gradient of the barrier is zero and then the algoirthm complains
                current_parameters_primary_controller["collision_load_"+str(kk)]     = 0
                
                current_parameters_secondary_controller["collision_pos_"+str(kk)]    = np.zeros((2,1))# here we put the value of the last agent found in the collsion radius, but it can be any nonzero random position since the constraint is not active
                current_parameters_secondary_controller["collision_switch_"+str(kk)] = 0
                current_parameters_secondary_controller["collision_load_"+str(kk)]     = 0
                
                
        # ---- SOLVING THE OPTIMIAZTION PROBLEM ----
        # solve the primary solver first and then try the secondat controller
        if self._warm_start_sol.size != 0    :
            try :
                sol = self._primary_solver( x0  =  self._warm_start_sol,
                                    p   =  current_parameters_primary_controller,
                                    ubg = 0)
                self._warm_start_sol   = sol['x'] # this is saved for warm starting the next optimization problem
        
            except Exception as e1:
                print(f"Agent {self._agent_id} Primary Controller Failed !")
                self.__LOGGER.error(f"Primary controller failed with the following message")
                self.__LOGGER.error(e1, exc_info=True)
                
                if self._enable_collision_avoidance :
                    self.__LOGGER.error(f"Triggering secondary controller.... ")
                    try :
                        sol = self._backup_solver( p   =  current_parameters_secondary_controller,
                                               ubg = 0)
                        
                    except Exception as e2:
                        print(f"Agent {self._agent_id} Secondary Controller Failed !")
                        self.__LOGGER.error(f"Secondary controller failed with the following message : {str(e2)}")
                        self.__LOGGER.error(e2, exc_info=True)
                        self.__LOGGER.error(f"Exiting...")
                        raise e2
                    
                else:
                    raise e1
        else :
            try :
                sol = self._primary_solver( p    =  current_parameters_primary_controller,
                                            ubg  = 0)
                self._warm_start_sol   = sol['x'] # this is saved for warm starting the next optimization problem
                
            except Exception as e1:
                print(f"Agent {self._agent_id} Primary Controller Failed !")
                self.__LOGGER.error(f"Primary controller failed with the following message")
                self.__LOGGER.error(e1, exc_info=True)
                
                if self._enable_collision_avoidance :
                    self.__LOGGER.error(f"Triggering secondary controller.... ")
                    try :
                        sol = self._backup_solver(  p   =  current_parameters_secondary_controller,
                                                ubg = 0)

                    except Exception as e2:
                        print(f"Agent {self._agent_id} Secondary Controller Failed !")
                        self.__LOGGER.error(f"Secondary controller failed with the following message : {str(e2)}")
                        self.__LOGGER.error(e2, exc_info=True)
                        self.__LOGGER.error(f"Exiting...")
                        raise e2
                else :
                    raise e1
                
        optimal_control_input  = sol['x'][:self._mathematical_model.control_input_sym.size1()] # extract the control input from the solution
        
        self.__LOGGER.info(f"Optimal control input computed successfully for agent {self._agent_id} : {optimal_control_input}")
        
        # resetting information for the controller
        self._best_impact_from_leaders   = {leader:None for leader in self._leader_neighbours} 
        self._gamma_tilde                = {leader:None for leader in self._leader_neighbours} 
        self._worse_impact_from_follower = None
        
        
        return optimal_control_input       
    
    
    #! : for Ernesto: This function could be done better using asynchrnous programming but I don't have the time to improve it now.
    #! : We can check together later
    def compute(self) -> tuple[np.ndarray,TimeSeries]:
        """
        Computes the control input for the agent based on the current state information of the agent and its neighbors.

        Args:
            current_neighbours_aw (list[StateInfo]): A list of StateInfo objects representing the current state information of the agent's neighbors.
            current_agent_aw (np.ndarray): An array representing the current state of the agent.
            current_time (float): The current time.

        Returns:
            optimal_control_input(np.ndarray): The optimal control input for the agent.
        """
        
        # continuously try to compute gamma based on the best impact from the leader agents
        self.request_best_impacts_from_leaders_and_compute_gamma_tilde_values()
        
        if self.is_ready_to_compute_gamma_and_best_impact and (not self._gamma_is_available) :
            self.compute_gamma_and_notify_best_impact_on_leader_task()
            self.compute_and_notify_worse_impact_on_tasks_you_are_following()
            self._gamma_is_available = True
        else :
            return (optimal_control_input,empty_intent)  #! just give a stand by command for the drone to stay in place
        
        if self._worse_impact_from_follower != None and self._gamma_is_available:
            self._gamma_is_available = False
            empty_intent = TimeSeries() # there is no intent from this controller. An intent could come from an MPC from example
            optimal_control_input = self.compute_optimal_control_input()
            return optimal_control_input,empty_intent
        
        
    
    def _update(self, control_input_and_intent: tuple[np.ndarray, TimeSeries]):
        """
        Update the agent's model with the computed control input
        and store the intent in the agent's awareness vector.

        Example
        -------
        A new controller could decide to override the default :meth:`_update` method.

        >>> from symaware.base import Controller, TimeSeries
        >>> class MyController(Controller):
        ...     def _update(self, control_input: np.ndarray, intent: TimeSeries):
        ...         # Your implementation here
        ...         # Example:
        ...         # Simply override the control input and intent of the agent
        ...         self._agent.model.control_input = control_input
        ...         self._agent.self_awareness.intent = intent

        Args
        ----
        control_input:
            New control input to apply to the agent's model
        intent:
            New intent to store in the agent's awareness vector
        """
        control_input, intent = control_input_and_intent
        


def new_reference_for_low_level_controller(agent: Agent, control_input, intent):
    """The result of the high level compute will be given here"""
    agent.controllers[1]._current_target_vx = control_input[0]
    agent.controllers[1]._current_target_vy = control_input[1]
    
### We will need to attach a callback on the agent to update the low level controller with the new reference
# For example : HighLevelController.add_on_computed(new_reference_for_low_level_controller)


class LowLevelController(Controller):
    
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
        try :
            I_yy = self._agent.model.I_yy
            I_xx = self._agent.model.I_xx
        except :
            raise RuntimeError("The agent model does not have the inertia values. Please check the model of the agent to be a drone")
        
        self._hoovering_force =   self._agent.model.g # misleading but it is saved like this in the drone model
    
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
        
        
    def compute(self) :
        """Simple PD controller to track a given velocity profile in the x-y component"""
        
        
        self._current_vx = self._agent.perception_system.current_agent_velocity[0]
        self._current_vy = self._agent.perception_system.current_agent_velocity[0]
        
        
        error_vx = self.current_target_vx  - self.current_vx
        error_vy = self.current_target_vy  - self.current_vy
        
        # check signs
        self.tau_theta = -self._Kp * error_vx - self._Kd * (error_vx - self._error_vx_prev)/self.async_loop_lock.time_interval
        self.tau_phi   = -self._Kp * error_vy - self._Kd * (error_vy - self._error_vy_prev)/self.async_loop_lock.time_interval
        
        self._error_vx_prev = error_vx
        self._error_vy_prev = error_vy
        
        torques = np.array([self.tau_phi,self.tau_theta, 0])
        force   = self._hoovering_force
        
        rpm = self._agent.model.convert_force_and_torque_to_rpm(force,torques)
        return rpm
    
    