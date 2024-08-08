"""
MIT License

Copyright (c) [2024] [Gregorio Marchesini]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import casadi as ca
from   typing import Callable
import logging
import time

from    stl.dynamics import MathematicalDynamicalModel
from    stl.stl import (StlTask,
                        IndependentLinearBarrierFunction,
                        CollaborativeLinearBarrierFunction,
                        IndependentSmoothMinBarrierFunction,
                        CollaborativeSmoothMinBarrierFunction,
                        IndependentPredicate,CollaborativePredicate, 
                        create_linear_barriers_from_task,
                        filter_tasks_by_time_limit,
                        SmoothMinBarrier,        
                        LinearBarrier,           
                        CollaborativeBarrierType,
                        IndependentBarrierType  ,
                        )

from data.data import (ControlMessage,ControlMessageType)

from   .utils import LeadershipToken
from   .optimization_module import BestImpactSolver, WorseImpactSolver

from symaware.base import (get_logger,
                           Controller,
                           MultiAgentAwarenessVector,
                           MultiAgentKnowledgeDatabase,
                           Agent,
                           TimeSeries)


class STLController(Controller):
    """STL-QP based controller"""
    def __init__(self, unique_identifier      : int,
                       dynamical_model_math   : MathematicalDynamicalModel,
                       look_ahead_time_window : float = float("inf")) -> None:
        
        
        """
        Args :
            unique_identifier (int) : unique identifier of the agent
            dynamical_model_math (DynamicalModel) : dynamical model of the agent
            log_level (int) : log level for the logger
            look_ahead_time_window (float) : time window to look ahead in the future

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        
        
        if not isinstance(dynamical_model_math,MathematicalDynamicalModel):
            raise ValueError(f"the dynamical model should be an instance of DynamicalModel. You have {type(dynamical_model_math)}")
            
        if dynamical_model_math.unique_identifier != unique_identifier:
            raise ValueError(f"the dynamical model unique_identifier is {dynamical_model_math.unique_identifier} but the agent unique_identifier is {unique_identifier}. They should be the same")
        
        if (look_ahead_time_window <= 0):
            raise ValueError(f"look_ahead_time_window should be a positive number. You have {look_ahead_time_window}")
        
        super().__init__()
        
        self._unique_identifier              = unique_identifier
        self._dynamical_model_math                = dynamical_model_math
        self._look_ahead_time_window         = look_ahead_time_window
        self._stl_tasks : list[StlTask]      = []    # list of all the tasks to be satisfied by the controller
        
        self._optimizer : ca.Opti            = ca.Opti() # optimization problem
        self._control_input_var : ca.MX      = self._optimizer.variable(2,1) # control input variable
        
        # information stored about your neighbours
        self._task_neighbours_id        : list[int]        = []    # list of neighbouring agents identifiers
        self._follower_neighbour        : int              = None   # this is the unique_identifier of the only agent that is your follower! (if you have one)
        self._leader_neighbours         : list[int]        = []     # this is the list of leader neighbours for the tasks around you
        
        self._best_impact_from_leaders   : dict[int,float] = dict()   # for each leader you will receive a best impact that you can use to compute your gamma
        self._worse_impact_on_leaders    : dict[int,float] = dict() # after you compute gamma you send back your worse impact to the leader
        
        self._worse_impact_on_leaders_stored_lambda : dict[int,Callable|None] = dict() # stores a simple lambda function that can be used to compute the worse impact when the gamma is computed
        
        self._worse_impact_from_follower : float|None = None # this is the worse impact that the follower of your task will send you back
        self._best_impact_on_follower    : float = 0. # this is the best impact that you can have on thhe task you are leading
        
        self._barrier_constraints         : list[ca.MX]                  = []    # barrier constraint
        self._barrier_functions           : dict[int,SmoothMinBarrier]   = {}    # for each neigbour you have a barrier function to satisfy
        
        self._slack_vars              : list[ca.MX]      = []
        self._warm_start_sol          : ca.OptiSol       = None
        self._initialization_time     : float            = None  # time of initialization for the simulation
        
        # computed control reduction factor gamma
        self._gamma                     : float            = 1      # 0 < gamma_i <= 1 control reduction function
        self._gamma_tilde               : dict[int,float|None]  = dict()     # list of all the intermediate gammas for each task. This corresponds to tilde_gamma_ij

        self._best_impact_solver  = BestImpactSolver(model = dynamical_model_math)
        self._worse_impact_solver = WorseImpactSolver(model = dynamical_model_math)
        
        scalar = ca.MX.sym("scalar",1)
        factor = 50
        self._alpha_fun = ca.Function("alpha_fun",[scalar],[ ca.if_else(scalar>=0,factor*scalar,0.002*scalar) ])
        
        # check if tasks are given to this controller
        self._has_tasks                  = False
        self._has_self_tasks             = False
        self._is_leaf                    = False
        self._has_already_computed_gamma = False
        self._last_velocity_input        = np.array([0.,0.]) 
        
        self.__LOGGER = get_logger(__name__, f"Controller {self._unique_identifier}")

    
    #exposed attributes according to main abstract class
    @property
    def unique_identifier(self) -> int:
        return self._unique_identifier
    @property
    def dynamical_model_math(self) -> MathematicalDynamicalModel:
        return self._dynamical_model_math
    @property
    def barrier_functions(self) -> dict[int,SmoothMinBarrier]:
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
    def is_ready_to_compute_gamma(self)-> bool :
        if self._is_leaf:
            return True
        else :
            return all([self._gamma_tilde[identifier] != None for identifier in self._leader_neighbours]) 
  
    @property
    def has_received_worse_impact_from_follower(self) -> bool:
        return self._worse_impact_from_follower != None
        
       
    def get_leader_and_follower_neighbours(self,leadership_tokens: dict[int,LeadershipToken]) -> tuple[list[int],int]:
        
        # Save all the leader neighbours
        leader_neighbours  = []
        follower_neighbour = None
        
        for unique_identifier,token in leadership_tokens.items() :
            if token == LeadershipToken.UNDEFINED :
                raise RuntimeError(f"The leadership token for agent {self._unique_identifier} is undefined. Please check that the token passing algorithm gives the correct result. No undefined tokens can be present")
            elif token == LeadershipToken.LEADER :
                follower_neighbour = unique_identifier # get the follower neighbour
            elif token == LeadershipToken.FOLLOWER :
                leader_neighbours.append(unique_identifier) # get the leader neighbours
        
        if len(leader_neighbours) == 0:
            self._is_leaf = True

            
        return leader_neighbours,follower_neighbour
       
        
        
    def check_task(self, task: StlTask) -> None:  
        """add tasks to the task list to be satisfied by the controller"""
        if isinstance(task, StlTask):
                if  not (self.unique_identifier in task.predicate.contributing_agents ) :
                    raise ValueError(f"Seems that one or more of the inserted tasks do not involve the state of the current agent. Indeed agent index is {self._unique_identifier}, but task is defined over agents ({task.predicate.contributing_agents})")
                else:
                    contributing_agents = task.predicate.contributing_agents
                    if len(contributing_agents) > 1:
                        self.__LOGGER.debug(f"Added tasks over edge : {task.predicate.contributing_agents}")
                    else:
                        self.__LOGGER.debug(f"Added self task : {[task.predicate.contributing_agents[0],task.predicate.contributing_agents[0]]}")
        else:
            raise Exception("please enter a valid STL task object or a a list of StlTask objects")
        
        
    def get_optimization_parameters_dictionary(self,task_neighbours) -> dict:
        
        params                = dict()
        params["state"]       = {}
        
        for neighbour in (task_neighbours + [self._unique_identifier]):
            params["state"][neighbour] = self._optimizer.parameter(2,1)
        
        params["state_closest_entity"]           = self._optimizer.parameter(2)
        params["time"]                           = self._optimizer.parameter(1)
        params["gamma"]                          = self._optimizer.parameter(1)
        params["worse_impact_from_follower"]     = self._optimizer.parameter(1)
        
        return params
    
    
    
    def get_barriers_from_tasks(self,tasks: list[StlTask], initial_conditions) -> dict[int,"SmoothMinBarrier"]:  
        
        barriers_storage = {}

        # For each edge, the predicate level set is divided into multiple linear barrier functions, which are store in a list.
        for task in tasks:
            if isinstance(task.predicate,CollaborativePredicate):
                new_barriers = create_linear_barriers_from_task( task = task,
                                                                     initial_conditions  = initial_conditions,
                                                                     t_init               = self._initialization_time,
                                                                     maximum_control_input_norm = self._dynamical_model_math.maximum_expressible_speed)
                
                if self._unique_identifier == task.predicate.source_agent:
                    barriers = barriers_storage.setdefault(task.predicate.target_agent,[]) 
                    barriers += new_barriers
                
                else :
                    barriers = barriers_storage.setdefault(task.predicate.source_agent,[]) 
                    barriers += new_barriers
                    
            elif isinstance(task.predicate,IndependentPredicate):
                new_barriers =  create_linear_barriers_from_task( task = task,
                                                                   initial_conditions         = initial_conditions,
                                                                   t_init                     = self._initialization_time,
                                                                   maximum_control_input_norm = self._dynamical_model_math.maximum_expressible_speed)
                barriers = barriers_storage.setdefault(self._unique_identifier,[])
                barriers += new_barriers 
                 
        
        # For each edge the barriers are compacted using the smooth minimum approximaiton
        smooth_min_barriers = { unique_identifier:None for unique_identifier in barriers_storage.keys()}
        
        # join the barriers in a single minimum approximation
        for identifier,barriers in  barriers_storage.items():
            
            eta_value = 40 # The higher the better, but the risk is to then run into numerical imprecision with the smooth min approximation
            if isinstance(barriers[0],CollaborativeLinearBarrierFunction):
                smooth_min_barriers[identifier] = CollaborativeSmoothMinBarrierFunction(list_of_barrier_functions=barriers, eta = eta_value) 
            elif isinstance(barriers[0],IndependentLinearBarrierFunction):
                smooth_min_barriers[identifier] = IndependentSmoothMinBarrierFunction(list_of_barrier_functions=barriers, eta = eta_value)
            else :
                raise ValueError(f"Barrier function should be either a CollaborativeLinearBarrierFunction or IndependentLinearBarrierFunction. You have {type(barriers[0])}. This error should not occur. Contact the developers")
                   
        return smooth_min_barriers
    
   
    def from_barrier_to_constraint(self,barrier: SmoothMinBarrier | LinearBarrier ,
                                         params: dict[str,ca.MX]) -> ca.MX :
        
        if isinstance(barrier,(CollaborativeSmoothMinBarrierFunction,CollaborativeLinearBarrierFunction) ):
            
            
            nabla_xi  :ca.MX  = barrier.gradient(agent_id = self._unique_identifier,
                                               x_source = params["state"][barrier._source_agent],
                                               x_target = params["state"][barrier._target_agent],
                                               t        = params["time"]) # gradient of the barrier w.r.t to agent i
            
            nabla_xi        = nabla_xi.reshape((1,nabla_xi.numel())) # enforce it to be a row
            
            gamma_dot       = barrier.time_derivative_at_time(x_source = params["state"][barrier._source_agent],
                                                              x_target = params["state"][barrier._target_agent],
                                                              t        =  params["time"])      # time derivative of the barrier computed at time params["time"]
            
            g_xi            = self._dynamical_model_math.g_fun(params["state"][self._unique_identifier]) # g(x) function of the agent (x_dot = f(x) + g(x)u
            
            barrier_value   = barrier.compute(x_source = params["state"][barrier._source_agent],
                                              x_target = params["state"][barrier._target_agent],
                                              t        = params["time"])
            
          
        
            # Determine which type of constraint should be considered for this specific case
            if self._unique_identifier == barrier._source_agent:
                neighbour_id = barrier._target_agent
            else:
                neighbour_id = barrier._source_agent
                
            # create the constraints 
            if neighbour_id in self._leader_neighbours: # they will take care of the rest of the barrier
                slack             = self._optimizer.variable(1)
                self._slack_vars += [slack]
                load_sharing      = 0.1
                
                barrier_constraint = -1* ( nabla_xi @ g_xi@self._control_input_var + load_sharing * (gamma_dot + self._alpha_fun(barrier_value)) + slack)
            
            elif neighbour_id == self._follower_neighbour: # No slack satisfaction here
                
                 barrier_constraint = -1* ( nabla_xi @ g_xi@self._control_input_var + (gamma_dot + self._alpha_fun(barrier_value)) + params["worse_impact_from_follower"])
            else :
                raise RuntimeError(f"Agent {self._unique_identifier} is not leading nor following agent {neighbour_id}. This is a bug. Contact the developers")
        
        
        elif isinstance(barrier, (IndependentSmoothMinBarrierFunction, IndependentLinearBarrierFunction) ):
            
            nabla_xi : ca.MX  = barrier.gradient(x=params["state"][self._unique_identifier],
                                        t=params["time"])
            nabla_xi          = nabla_xi.reshape((1,nabla_xi.numel())) # enforce it to be a row
            
            gamma_dot = barrier.time_derivative_at_time(x=params["state"][self._unique_identifier],
                                                        t=params["time"])
            g_xi = self._dynamical_model_math.g_fun(params["state"][self._unique_identifier])
            
            barrier_value = barrier.compute(x=params["state"][self._unique_identifier],
                                            t=params["time"])
            
            if self._follower_neighbour == None: # In this case you are the global leader
                barrier_constraint = -1* ( nabla_xi @ g_xi@self._control_input_var + (gamma_dot + self._alpha_fun(barrier_value)))
            else :
                slack              = self._optimizer.variable(1)
                self._slack_vars  += [slack]
                barrier_constraint = -1* ( nabla_xi @ g_xi@self._control_input_var + (gamma_dot + self._alpha_fun(barrier_value)) + slack)
        
        else :
            raise ValueError(f"Barrier function should be either a CollaborativeSmoothMinBarrierFunction, CollaborativeLinearBarrierFunction, IndependentSmoothMinBarrierFunction or IndependentLinearBarrierFunction. You have {type(barrier)}")
        return barrier_constraint 
        
        
    def generate_barrier_constraints(self,tasks: list[StlTask], initial_conditions) -> list[ca.MX] :
        """This create a distributed barrier constraint from a barrier funciton. Hence if you have sum_i \nabla_xi b (f_i+g_i*u_1) + db_dt + alpha(b) >=0 yhen the angent will only have the constraint \nabla_xi b (f_i+g_i*u_1) + loading_factor(db_dt + alpha(b)) >=0"""
        
        constraints = []
        self._barrier_functions = self.get_barriers_from_tasks(tasks,initial_conditions)
        for barrier in self._barrier_functions.values():
            constraints += [self.from_barrier_to_constraint(barrier,self._params)]
        
        return constraints
    
    def get_slack_constraints(self) -> list[ca.MX]:
        
        return [-slack  for slack in self._slack_vars]
    
    def get_control_input_constraints(self) -> list[ca.MX]:
        
        A_u = self._dynamical_model_math.input_constraints_A
        b_u = self._dynamical_model_math.input_constraints_b
        
        return [A_u @ self._control_input_var - b_u*self._params["gamma"]]
    
    
    def collision_avoidance_constraint(self) -> ca.MX:
        
        relative_pos         = self._params["state"][self._unique_identifier] - self._params["state_closest_entity"]
        alpha_coefficient    = 1
        collision_rad        = 0.01
        collision_constraint = -1*(relative_pos.T@(self._control_input_var) + alpha_coefficient/2 * (ca.sumsqr(relative_pos)-collision_rad**2)) # works assuming the other agent is also trying to avoid you 
        
        return collision_constraint
    
    
    
    def compute_gamma_tilde_values(self, agents_state  : dict[int,np.ndarray]     ,
                                         current_time  : float):
        
        if self._is_leaf:
            self.__LOGGER.debug(f"agent is leaf, no need to compute gammas...")
            return 
            
        self.__LOGGER.debug(f"Trying to compute gamma tilde values...")
        
        for neighbour in self._leader_neighbours:
            barrier = self._barrier_functions[neighbour]
            
            if self._gamma_tilde[neighbour] == None: # try to compute it
                gamma_tilde = self._compute_gamma_tilde_for_barrier(barrier,agents_state,current_time)
            else :
                continue # just go to the next one if you have already computed it
            
            self._gamma_tilde[neighbour] = gamma_tilde
                        
    
    def _compute_gamma_tilde_for_barrier(self,barrier: CollaborativeBarrierType , agents_states :dict[int,np.ndarray],current_time: float) -> float :
        
        # NOTE: This function should only be called when the barrier neighbour is among the leader neighbours
        
        
        if barrier.source_agent == self._unique_identifier:
            neighbour_id = barrier.target_agent
        else :
            neighbour_id = barrier.source_agent
        
        try :
            neighbour_best_impact  = self._best_impact_from_leaders[neighbour_id]
        except KeyError:
            raise RuntimeError(f"Agent {self._unique_identifier} does not have the best impact from leader {neighbour_id} bceause this is not a leader for him. This is a bug. Contact the developers")
        
        if neighbour_best_impact is None: # Best impact is not available yet!
            return None
                
        current_agent_state  = agents_states[self._unique_identifier]  # your current state
        nabla_xi :np.ndarray = barrier.gradient(agent_id = self._unique_identifier,
                                                x_source = agents_states[barrier.source_agent],
                                                x_target = agents_states[barrier.target_agent],
                                                t        = current_time)
        if isinstance(nabla_xi,ca.DM):
            nabla_xi        = nabla_xi.full()
        
        nabla_xi        = nabla_xi.flatten()
        barrier_value   = barrier.compute(x_source = agents_states[barrier.source_agent],
                                          x_target = agents_states[barrier.target_agent],
                                          t        = current_time)
        self.__LOGGER.debug(f"Gradient for the barrier {(barrier.source_agent,barrier.target_agent)} :  {nabla_xi}")    
        self.__LOGGER.debug(f"Value of the barrier {(barrier.source_agent,barrier.target_agent)} : {barrier_value}")
        self.__LOGGER.debug(f"Current state of the agent {self._unique_identifier} : {current_agent_state}")
        self.__LOGGER.debug(f"Current state of neighbour {neighbour_id} : {agents_states[neighbour_id]}")
        
        
        if np.linalg.norm(nabla_xi) <= 1E-6 : # case when the gradient is practically zero
            gamma_tilde = 1
            self._worse_impact_on_leaders_stored_lambda[neighbour_id]  = lambda gamma : 0 # once multipleied for gamma becaomse again (Lf + Lg@ worse_input*gamma)
            return gamma_tilde
        
        else : # compute the gamma tilde value
            g_value : np.ndarray = self._dynamical_model_math.g_fun(current_agent_state).full()
            f_value : np.ndarray = self._dynamical_model_math.f_fun(current_agent_state).full()
            
            
            # this can be parallelised for each of the barrier you are a follower of
            Lg = nabla_xi @ g_value
            Lf = nabla_xi @ f_value
            
            worse_input = self._worse_impact_solver.compute(Lg).full()
            self._worse_impact_on_leaders_stored_lambda[neighbour_id]  = lambda gamma : (Lf/(gamma+1E-5) + Lg@ worse_input)*gamma # once multipleied for gamma becaomse again (Lf + Lg@ worse_input*gamma)
                
            barrier_value        = barrier.compute(x_source = agents_states[barrier.source_agent],
                                                    x_target = agents_states[barrier.target_agent],
                                                    t        = current_time)
            
            self.__LOGGER.debug(f"value of the barrier is {barrier_value}")
            
            alpha_barrier_value = self._alpha_fun(barrier_value)
            time_derivative     = barrier.time_derivative_at_time(x_source = agents_states[barrier.source_agent],
                                                                    x_target = agents_states[barrier.target_agent],
                                                                    t        = current_time)
            
            if alpha_barrier_value < 0 :
                self.__LOGGER.warning(f"Barrier value is negative : {barrier_value}.")
            
            zeta        = alpha_barrier_value + time_derivative   
            gamma_tilde =  -(neighbour_best_impact + zeta + Lf) / ( Lg @ worse_input) # compute the gamma value    
            
            if gamma_tilde<= 0.:
                self.__LOGGER.info(f"Registered a gamma tilde less than zero: the alpha(barrier) is {alpha_barrier_value}," +
                                                                              f"the time derivative is {time_derivative}," +
                                                                              f"the best impact is {neighbour_best_impact}," +
                                                                              f"the value of the barrier is {barrier_value}," +
                                                                              f"the value of the gradient (it is the same for the leader just opposite sign) is {nabla_xi}," +
                                                                              f"the value of the g function is {g_value}," + 
                                                                              f"the value of the f function is {f_value}," +
                                                                              f"the value of the worse impact is {worse_input}, the neighbour id is {neighbour_id}.")
            
            return float(gamma_tilde)
    
        
    def compute_gamma(self) :
        self.__LOGGER.debug(f"Computing gammas")
        
        # Now it is the time to check if you have all the available information
        if self._is_leaf : 
            self._gamma = 1
        
        if self.is_ready_to_compute_gamma :
            self._gamma = min( list(self._gamma_tilde.values()) + [1]) # take the minimum of the gamma tilde values
            if self._gamma<=0 :
                self.__LOGGER.info(f"The computed gamma value is negative with value {self._gamma}. The gamma value is set at zero.")
                self._gamma = 0
        
        self.__LOGGER.debug(f"Gamma value is {self._gamma}")
        
    def compute_best_impact_for_follower(self,agents_states:dict[int,np.ndarray],current_time: float):
        
        self.__LOGGER.debug(f"Computing and notifying best impact for follower {self._follower_neighbour}")
        # If you have no follower neighbour then you can just end the process
        if self._follower_neighbour == None :
            return
        
        barrier  :CollaborativeSmoothMinBarrierFunction = self._barrier_functions[self._follower_neighbour]
        current_agent_state  = agents_states[self._unique_identifier]  # your current state
        nabla_xi :np.ndarray = barrier.gradient(agent_id = self._unique_identifier,
                                                x_source = agents_states[barrier.source_agent],
                                                x_target = agents_states[barrier.target_agent],
                                                t        = current_time)
        self._gradient_leader_task = nabla_xi
        if isinstance(nabla_xi,ca.DM):
            nabla_xi        = nabla_xi.full()
        
        nabla_xi             = nabla_xi.flatten()
        g_value : np.ndarray = self._dynamical_model_math.g_fun(current_agent_state).full()
        f_value : np.ndarray = self._dynamical_model_math.f_fun(current_agent_state).full()    

        # this can be parallelised for each of the barrier you are a follower of
        Lg = nabla_xi @ g_value
        Lf = nabla_xi @ f_value
        
        if np.linalg.norm(Lg) <= 1E-6 :
            self._best_impact_on_follower = Lf
        else :
            best_impact = self._best_impact_solver.compute(Lg).full()
            self._best_impact_on_follower   = Lf + Lg @ best_impact*self._gamma  
         
        
        message = ControlMessage(type = ControlMessageType.BEST_IMPACT,
                                 time_stamp = current_time,
                                 value = self._best_impact_on_follower,
                                 sender_id=self._unique_identifier,
                                 receiver_id = self._follower_neighbour)
        
        self._agent.communication_sender.enqueue_messages(message)
        self._agent.communication_sender.async_loop_lock.trigger()
        
    
    def compute_worse_impact_for_leaders(self):
        for leader in self._leader_neighbours:
            worse_impact = self._worse_impact_on_leaders_stored_lambda[leader](self._gamma)
            self._worse_impact_on_leaders[leader] = worse_impact
            message = ControlMessage(type       = ControlMessageType.WORSE_IMPACT,
                                    time_stamp  = time.time(),
                                    value       = worse_impact,
                                    sender_id   = self._unique_identifier,
                                    receiver_id = leader)
            self._agent.communication_sender.enqueue_messages(message)
        
        self._agent.communication_sender.async_loop_lock.trigger()
            
            
        
    
    def compute_control_input(self, current_states:dict[int,np.ndarray], current_time: float) -> np.ndarray:
        
        # gamma equal to zero just stand by
        if self._gamma == 0:
            self.__LOGGER.info(f"Value of gamma is zero. stand by")
            self.flush_current_information()
            return np.zeros((2,1))
        
        # to be adapted when dealing with real time implementation
        if (self._worse_impact_from_follower == None) and (self._follower_neighbour != None):
            self.__LOGGER.error(f"The worse impact from the follower is not computed. Follower neighbour is {self._follower_neighbour}")
            raise RuntimeError(f"The worse impact from the follower is not computed.Follower neighbour is {self._follower_neighbour}")
        
        
        # set values of the parameters for the controller
        for id in self._params["state"].keys():
            self._optimizer.set_value(self._params["state"][id], current_states[id])
            
        self._optimizer.set_value(self._params["state_closest_entity"],self.get_closest_state(current_states) )
        self._optimizer.set_value(self._params["time"], current_time)
        self._optimizer.set_value(self._params["gamma"], self._gamma)
        if self._follower_neighbour != None:
            self._optimizer.set_value(self._params["worse_impact_from_follower"],self._worse_impact_from_follower)

        failed_computation = False
        if self._warm_start_sol == None    :
            try :
                sol = self._optimizer.solve()
                self._warm_start_sol   = sol
        
            except Exception as e1:
                self.__LOGGER.info(f"Primary controller did not find a solution. Stand by.... The value of gamma is : {self._gamma}")
                # self.__LOGGER.error(e1, exc_info=True)
                failed_computation = True
                
        else :
            try :
                self._optimizer.set_initial(self._warm_start_sol.value_variables()) # if you get an error named "is_regular() failed" it means that the warm start solution had some nans. 
                                                                                    # This comes from the fact that some variables are in the problem but are not in the objective or constraints (so they are returned as NAN in the solution)
                sol = self._optimizer.solve()
                self._warm_start_sol   = sol
        
            except Exception as e2:
                self.__LOGGER.info(f"Primary controller did not find a solution. Stand by.... The value of gamma is : {self._gamma}")
                # self.__LOGGER.error(e2, exc_info=True)
                failed_computation = True
        
        # flush data.
        self.flush_current_information()
        
        if failed_computation:
            failed_computation = False
            return np.zeros((2,1))
        else :
            return sol.value(self._control_input_var)
        
        
    def flush_current_information(self) :
        # flushing old information
        self._best_impact_from_leaders              = {unique_identifier : None for unique_identifier in self._leader_neighbours} 
        self._gamma_tilde                           = {unique_identifier : None for unique_identifier in self._leader_neighbours}  
        self._worse_impact_from_follower            = None
        self._best_impact_on_follower               = 0.
        self._worse_impact_on_leaders_stored_lambda = {unique_identifier : None for unique_identifier in self._leader_neighbours}  
        self._worse_impact_on_leaders               = {unique_identifier : None for unique_identifier in self._leader_neighbours}
    
    def get_closest_state(self,agents_states:dict[int,np.ndarray]) -> np.ndarray:
        """This function is used to get the state of the closest agent to the current agent"""
        
        current_state  = agents_states[self._unique_identifier] 
        
        others        = [state for id,state in agents_states.items() if id != self._unique_identifier]
        distances     = [np.linalg.norm(current_state - state) for state in  others]
        closest_state = others[np.argmin(distances)]
        
        return closest_state
    
    
    # todo: set up these functions correctly
    def initialise_component(
        self,
        agent: "Agent",
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase,
    ):
        
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        
        # extract position
        initial_conditions = {unique_identifier : initial_awareness_database[unique_identifier].state[:2] for unique_identifier in initial_awareness_database.keys()}
        stl_tasks          = initial_knowledge_database[self._unique_identifier]["stl_tasks"]
        initial_time       = initial_knowledge_database[self._unique_identifier]["initial_time"]
        leadership_tokens  = initial_knowledge_database[self._unique_identifier]["leadership_tokens"]
        
        """Used to set up the controller. The main objective here it is to transform the tasks into barriers and then to initialise the constraints for the optimization problem"""
        self.__LOGGER.debug(f"Setting up the controller")
        
        # Save all the leader neighbours
        self.__LOGGER.debug(f"Recorded leadreship tokens : {leadership_tokens}")
        self._leader_neighbours , self._follower_neighbour = self.get_leader_and_follower_neighbours(leadership_tokens)
        self._best_impact_from_leaders = {unique_identifier : None for unique_identifier in self._leader_neighbours} # initialise the best impact from leaders
        self._task_neighbours_id       = self._leader_neighbours + [self._follower_neighbour] if self._follower_neighbour is not None else self._leader_neighbours
     
        self._initialization_time = initial_time  # time at which the problem is initialized
        self._params              = self.get_optimization_parameters_dictionary(self._task_neighbours_id)   
        self._gamma_tilde         = {unique_identifier : None for unique_identifier in self._leader_neighbours} # initialise the gamma tilde values 

     
        # add the tasks to the controller
        try :
            stl_tasks = iter(stl_tasks)
        except TypeError:
            raise TypeError("stl_tasks should be an iterable of StlTask objects")
        stl_tasks = filter_tasks_by_time_limit(stl_tasks,initial_time,initial_time + self._look_ahead_time_window)
        for task in stl_tasks:
            self.check_task(task) # raises an error in case the task is not correct
            self._stl_tasks.append(task)
        
        # now get all the constraints for the optimization problem
        self._barrier_constraints      = self.generate_barrier_constraints(self._stl_tasks,initial_conditions)
        slack_constraints              = self.get_slack_constraints()
        control_input_constraints      = self.get_control_input_constraints()
        collision_avoidance_constraint = self.collision_avoidance_constraint()
        
        self._optimizer.subject_to(ca.vertcat(*self._barrier_constraints) <= 0)
        self._optimizer.subject_to(ca.vertcat(*control_input_constraints) <= 0)
        # self._optimizer.subject_to(collision_avoidance_constraint <= 0) #todo! : check here when you can add the collision avoidance constraint again
        if len(slack_constraints) != 0:
            self._optimizer.subject_to(ca.vertcat(*slack_constraints) <= 0)
        
        # set up cost
        cost  = self._control_input_var.T @  self._control_input_var # classic quadratic cost (work for simple integrator types of system)
        cost += ca.vertcat(*self._slack_vars).T @ ca.vertcat(*self._slack_vars) # add slack variables to the cost
        
        
        p_opts = dict(print_time=False, 
                        verbose=False,
                        expand=True)

        
        # Solver options
        s_opts = dict(
            print_level=0,
            tol=1e-6,
            max_iter=1000,
            )
        
        self._optimizer.minimize(cost)
        self._optimizer.solver("ipopt",p_opts,s_opts)
        self.__LOGGER.debug(f"Recorded Follower neighbour : {self._follower_neighbour}")
        
    

    def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        """
        This objective of this controller it is to pass an high level velocity command to 
        the agent. The agent has to wait to have received the best impact from the follower agent before computing its control input.
        So at each iteration this component only sends message to the other agents until it is ready to send an updated velocity command to the low level controller. 


        Returns
        -------
            - Most recent control input 
            - empty time series
        """
        
        agents_states = {unique_identifier : self._agent.self_awareness.state[:2] for unique_identifier in self._agent.self_awareness.state.keys()}
        current_time  =  time.ctime() - self._initialization_time
        
        if not self.is_ready_to_compute_gamma:
            self.compute_gamma_tilde_values(current_time =  current_time, 
                                            agents_state = agents_states)
        
        if self.is_ready_to_compute_gamma and (not self._has_already_computed_gamma):
            self.compute_gamma()
            self.compute_best_impact_for_follower(agents_states = agents_states,
                                              current_time  = current_time)
            self.compute_worse_impact_for_leaders()
            self._has_already_computed_gamma = True
        
        
        if self.has_received_worse_impact_from_follower:
            self._has_already_computed_gamma = False
            self._last_velocity_input = self.compute_control_input(agents_states,current_time) # update control input
            
        
        return (self._last_velocity_input,TimeSeries())
        

    def _update(self, control_input_and_intent: tuple[np.ndarray, TimeSeries]):
        """
        Notify low level controller of new control input and store new intent in awareness vector

        Args
        ----
        control_input:
            New control input to apply to the agent's model
        intent:
            New intent to store in the agent's awareness vector
        """
        control_input, intent = control_input_and_intent
        
        if control_input is not None:
            # notify the low-level controller of a new control input
            self._notify("new_control_input",control_input)
        self._agent.self_awareness.intent = intent # not really doing anything but kept here