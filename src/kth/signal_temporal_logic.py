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
from   typing import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass

from  kth.dynamics import StateName
from  kth.dynamics import DynamicalModel
from typing import TypeAlias

UniqueIdentifier : TypeAlias = int # identifier of a single agent in the system


# some support functions    

def first_word_before_underscore(string: str) -> str:
    """split a string by underscores and return the first element"""
    return string.split("_")[0]


def check_barrier_function_input_names(barrier_function: ca.Function)-> bool:
    for name in barrier_function.name_in():
        if not first_word_before_underscore(name) in ["state","time"]:
            return False
    return True    

def check_barrier_function_output_names(barrier_function: ca.Function)->bool:
    for name in barrier_function.name_out():
        if not first_word_before_underscore(name) == "value":
            return False
    return True

def is_time_state_present(barrier_function: ca.Function) -> bool:
    return "time" in barrier_function.name_in() 


def check_barrier_function_IO_names(barrier_function: ca.Function) -> bool:
    if not check_barrier_function_input_names(barrier_function) :
         raise ValueError("The input names for the predicate functons must be in the form 'state_i' where ''i'' is the agent ID and the output name must be 'value', got input nmaes " + str(function.name_in()) + " and output names " + str(function.name_out()) + " instead")
    
    elif not is_time_state_present(barrier_function) :
        raise ValueError("The time variable is not present in the input names of the barrier function. PLease make sure this is a function of time also (even if time could be not part of the barrier just put it as an input)")
    elif not check_barrier_function_output_names(barrier_function) :
        raise ValueError("The output name of the barrier function must be must be 'value'")
    

def check_predicate_function_input_names(predicate_function: ca.Function)-> bool:
    for name in predicate_function.name_in():
        if not first_word_before_underscore(name) in ["state"]:
            return False
    return True    


def check_predicate_function_output_names(predicate_function: ca.Function)->bool:
    for name in predicate_function.name_out():
        if not first_word_before_underscore(name) == "value":
            return False
    return True


def check_predicate_function_IO_names(predicate_function: ca.Function) -> bool:
    return check_predicate_function_input_names(predicate_function) and check_predicate_function_output_names(predicate_function)


def state_name_str(agent_id: UniqueIdentifier) -> str:
    """_summary_

    Args:
        agent_id (UniqueIdentifier): _description_

    Returns:
        _type_: _description_
    """    
    return f"state_{agent_id}"

def get_id_from_input_name(input_name: str) -> UniqueIdentifier:
    """Support function to get the id of the agents involvedin the satisfaction of this barrier function

    Args:
        input_names (list[str]): _description_

    Returns:
        list[UniqueIdentifier]: _description_
    """    
    if not isinstance(input_name,str) :
        raise ValueError("The input names must be a string")
    
 
    splitted_input_name = input_name.split("_")
    if 'state' in splitted_input_name :
        ids = int(splitted_input_name[1])
    else :
        raise RuntimeError("The input name must be in the form 'state_i' where ''i'' is the agent ID")
    
    return ids
    

class PredicateFunction :
    """
    PredicateFunction definition class. This class is used to store the information about a predicate function. The class is used to store the predicate function and the contributing agents to the satisfaction of the predicate function.
    """
    def __init__(self,
                 function: ca.Function) -> None:
        

        if not isinstance(function, ca.Function):
            raise TypeError("function must be a casadi.MX object") 
        
        if not check_predicate_function_IO_names(function) :
            raise ValueError("The input names for the predicate functons must be in the form 'state_i' where ''i'' is the agent ID and the output name must be 'value', got input names " + str(function.name_in()) + " and output names " + str(function.name_out()) + " instead")
        
        self._function = function
        self._contributing_agents = [get_id_from_input_name(name) for name in function.name_in()]   
    
    @property
    def function(self) :
        return self._function
    @property
    def contributing_agents(self):
        return self._contributing_agents
    


class BarrierFunction:
    """class for convex barrier functions"""
    def __init__(self,
                 function: ca.Function,
                 associated_alpha_function:ca.Function = None,
                 time_function:ca.Function = None,
                 switch_function:ca.Function = None,
                 name :str = None) -> None:
        
        """
        The initialization for a barrier function is a function b("state_1","state_2",...., "time")-> ["value].
        This type of structure it is checked within the class itself. The associated alpha funnction is is a scalar functon that can be used to 
        construct a barrie constraint in the form \dot{b}("state_1","state_2",...., "time") <= alpha("time")
        
        Args : 
            function                  (ca.Function) : the barrier function
            associated_alpha_function (ca.Function) : the associated alpha function
            switch_function           (ca.Function) : simple function of time that can be used to activate and deactivate the barrier. Namely the value is 1 if t<= time_of_remotion and 0 otherwise. Time is assumed to start from 0
        
        Example:
        >>> b = ca.Function("b",[state1,state2,time],[ca.log(1+ca.exp(-state1)) + ca.log(1+ca.exp(-state2))],["state_1","state_2","time"],["value"])
        >>> alpha = ca.Function("alpha",[dummy_scalar],[2*dummy_scalar])
        >>> barrier = BarrierFunction(b,alpha)
        
        """

        if not isinstance(function, ca.Function):
            raise TypeError("function must be a casadi.MX object") 
        
        check_barrier_function_IO_names(function) # will thraw an exception if the wrong naming in input and output is given

        self._function :ca.Function= function
        self._switch_function = switch_function
        self.check_that_is_scalar_function(function=associated_alpha_function) # throws an error if the given function is not valid one
        self._associated_alpha_function = associated_alpha_function
        self.check_that_is_scalar_function(function=time_function) # throws an error if the given function is not valid one
        self._time_function = time_function
        
        
        names = [name for name in function.name_in() if name != "time"] # remove the time
        self._contributing_agents = [get_id_from_input_name(name) for name in names] 
        self._gradient_function_wrt :dict[UniqueIdentifier,ca.Function] = {}
        
        
        self._partial_time_derivative :ca.Function = ca.Function()
        self._compute_local_gradient_functions() # computes local time derivatives and gradient functions wrt the state of each agent involved in the barrier function
        
        if name == None :
            self._name = self._function.name()
    
    
    @property
    def function(self) :
        return self._function
    @property
    def name(self):
        return self._name
    
    @property
    def associated_alpha_function(self):
        return self._associated_alpha_function
    

    @property
    def partial_time_derivative(self):
        return self._partial_time_derivative 

    @property
    def contributing_agents(self):
        return self._contributing_agents
    
    @property
    def switch_function(self):
        return self._switch_function
    @property
    def time_function(self):
        return self._time_function

    def gradient_function_wrt_state_of_agent(self,agent_id:UniqueIdentifier) -> ca.Function:
        try :
            return self._gradient_function_wrt[agent_id]
        except KeyError :
            raise KeyError("The gradient function with respect to agent " + str(agent_id) + " is not stored in this barrier function")
    
    # this function is applicable to general barriers
    def _compute_local_gradient_functions(self) -> None:
        """store the local gradient of the barrier function with respect to the given agent id. The stored gradienjt function takes as input the same names the barrier function"""
        
        named_inputs : dict[str,ca.MX]  = {} # will contain the named inputs to the function
        input_names  : list[str]        = self._function.name_in()

        for input_name in input_names :
            variable = ca.MX.sym(input_name,self._function.size1_in(input_name)) # create a variable for each state
            named_inputs[input_name] = variable
        
        for input_name in input_names :

            if first_word_before_underscore(input_name) == "state":
                state_var = named_inputs[input_name]
                
                nabla_xi  = ca.jacobian(self._function.call( named_inputs)["value"] , state_var) # symbolic gradient computation
                state_id  = get_id_from_input_name(input_name)
                self._gradient_function_wrt[state_id] = ca.Function("nabla_x"+str(state_id),list(named_inputs.values()), [nabla_xi],input_names,["value"]) # casadi function for the gradient computation
            
            elif input_name == "time" :
                time_variable                 = named_inputs[input_name]
                partial_time_derivative       = ca.jacobian(self._function.call(named_inputs)["value"],time_variable) # symbolic gradient computation
                self._partial_time_derivative = ca.Function("local_gradient_function",list(named_inputs.values()), [partial_time_derivative],list(named_inputs.keys()),["value"]) # casadi function for the gradient computation
    
    
    def check_that_is_scalar_function(self,function:ca.Function|None) -> None :
        
        if function == None :
            pass
        else: 
            if not isinstance(function,ca.Function) :
                raise TypeError("The function must be a casadi function")
            if function.n_in() != 1 :
                raise ValueError("The  function must be a scalar function of one variable")
            if not  function.size1_in(0) == 1 :
                raise ValueError("The  function must be a scalar function of one variable")


# Temporal Operators 
class TimeInterval :
    """ Time interval class to represent time intervals in the STL tasks"""
    def __init__(self,a:float|None = None,b:float|None =None) -> None:
        
        
        if any([a==None,b==None]) and (not all(([a==None,b==None]))) :
            raise ValueError("only empty set is allowed to have None Values for both the extreems a and b of the interval. Please revise your input")
        elif  any([a==None,b==None]) and (all(([a==None,b==None]))) : # empty set
            self._a = a
            self._b = b
        else :    
            if a>b :
                raise ValueError("Time interval must be a couple of non decreasing time instants")
            
            try :
                self._a = float(a)
                self._b = float(b)
            except :
                raise ValueError(f"The given time instants must be convertible to float. Given types are {type(a)}  and {type(b)}")
        
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b
    
    @property
    def measure(self) :
        if self.is_empty() :
            return None # empty set has measure None
        return self._b - self._a
    
    @property
    def aslist(self) :
        return [self._a,self._b]
    
    def is_empty(self)-> bool :
        if (self._a == None) and (self._b == None) :
            return True
        else :
            return False
        
    def is_singular(self)->bool:
        a,b = self._a,self._b
        if a==b :
            return True
        else :
            return False
        
        
        
    def __truediv__(self,timeInt:"TimeInterval") -> "TimeInterval" :
        """returns interval Intersection"""
        
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        
        # If any anyone is empty return the empty set.
        if self.is_empty() or timeInt.is_empty() :
            return TimeInterval(a = None, b = None)
        
        # check for intersection.
        if b2<a1 :
            return TimeInterval(a = None, b = None)
        elif a2>b1 :
            return TimeInterval(a = None, b = None)
        else :
            return TimeInterval(a = max(a2,a1), b = min(b1,b2))
            
        
    
    def __eq__(self,timeInt:"TimeInterval") -> bool:
        """ equality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return True
        else :
            return False
    
    def __ne__(self,timeInt:"TimeInterval") -> bool :
        """ inequality check """
        a1,b1 = timeInt.a,timeInt.b
        a2,b2 = self._a,self._b
        
        if a1 == a2 and b1 == b2 :
            return False
        else :
            return True
        
        
    
    def __lt__(self,timeInt:"TimeInterval") -> "TimeInterval":
        """strict subset relations self included in timeInt ::: "TimeInterval" < timeInt """
        
        intersection = self.__truediv__(timeInt)
        
        if intersection.is_empty() :
            return False
        
        if (intersection!=self) :
            return False
        
        if (intersection == self) and (intersection != timeInt) :
            return True
        else :
            return False # case of equality between the sets
    
        
    def __le__(self,timeInt:"TimeInterval") -> "TimeInterval" :
        """subset (with equality) relations self included in timeInt  ::: "TimeInterval" <= timeInt """
        
        intersection = self.__truediv__(timeInt)
        
        if intersection.is_empty() :
            return False
        
        if (intersection!=self) :
            return False
        
        if (intersection == self) :
            return True
   
        
        
    def __str__(self):
        return f"[{self.a},{self.b}]"
    

    def getCopy(self) :
        return TimeInterval(a = self.a, b=self.b)
            
      
# definition of the main temporal operators 



class TemporalOperator(ABC):
    
    
    @property
    @abstractmethod
    def time_of_satisfaction(self) -> float:
        pass
    
    @property
    @abstractmethod
    def time_of_remotion(self) -> float:
        pass
    

class AlwaysOperator(TemporalOperator):
    def __init__(self,time_interval:TimeInterval) -> None:
        self._time_interval         : TimeInterval = time_interval
        self._time_of_satisfaction   : float        = self._time_interval.a
        self._time_of_remotion      : float        = self._time_interval.b
    
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
    @property
    def time_interval(self) -> TimeInterval:
        return self._time_interval
    
    
class EventuallyOperator(TemporalOperator):
    def __init__(self,time_interval:TimeInterval,time_of_satisfaction:float=None) -> None:
        """ Eventually operator
        Args:
            time_interval (TimeInterval): time interval that referes to the eventually operator
            time_of_satisfaction (float): time at which the eventually operatopr is satisfied

        Raises:
            ValueError: _description_
        """
        self._time_interval       : TimeInterval = time_interval
        self._time_of_satisfaction : float       = time_of_satisfaction
        self._time_of_remotion    : float        = self._time_interval.b
        
        if time_of_satisfaction == None :
            self._time_of_satisfaction = time_interval.a + np.random.rand()*(time_interval.b- time_interval.a)
            
        elif time_of_satisfaction<time_interval.a or time_of_satisfaction>time_interval.b :
            raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{time_interval.a},{time_interval.b}]")
        
    
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
    @property
    def time_interval(self) -> TimeInterval:
        return self._time_interval

#! to be analysed again
class AlwaysEventuallyOperator(TemporalOperator):
    def __init__(self,always_time_interval:TimeInterval,eventually_time_interval:TimeInterval,eventually_time_of_satisfaction:float=None) -> None:
        
        #pick it random if not given
        if eventually_time_of_satisfaction == None :
            eventually_time_of_satisfaction = eventually_time_interval.a + np.random.rand()*(eventually_time_interval.b- eventually_time_interval.a) # random time of satisfaction
        else :
            if eventually_time_of_satisfaction<eventually_time_interval.a or eventually_time_of_satisfaction>eventually_time_interval.b :
                raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{eventually_time_interval.a},{eventually_time_interval.b}]")
        
        self._period               : TimeInterval = eventually_time_of_satisfaction # from the point "a" of the evetually, we have that the task is satisfied everty evetually_time_of_satsifaction
        
        self._time_of_satisfaction : float       = always_time_interval.a + eventually_time_interval.a # we satisfy at the initial time of the always first
        
        self._time_of_remotion     : float       = self._time_of_satisfaction +  np.ceil((always_time_interval.b - self._time_of_satisfaction)/ self._period) * self._period

        
    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
   
    @property
    def period(self) -> TimeInterval:
        return self._period

#! to be analysed again
class EventuallyAlwaysOperator(TemporalOperator):
    def __init__(self,always_time_interval:TimeInterval,eventually_time_interval:TimeInterval,eventually_time_of_satisfaction:float=None) -> None:
        
        
        if eventually_time_of_satisfaction == None :
             self._time_of_satisfaction = eventually_time_interval.a + np.random.rand()*(eventually_time_interval.b- eventually_time_interval.a) # random time of satisfaction
        else :
            if eventually_time_of_satisfaction<eventually_time_interval.a or eventually_time_of_satisfaction>eventually_time_interval.b :
                raise ValueError(f"For eventually formulas you need to specify a time a satisfaction for the formula in the range of your time interval [{eventually_time_interval.a},{eventually_time_interval.b}]")
            
        self._time_of_satisfaction : float    = eventually_time_of_satisfaction
        self._time_of_remotion     : float    = self._time_of_satisfaction  + always_time_interval.period

    @property
    def time_of_satisfaction(self) -> float:
        return self._time_of_satisfaction
    
    @property
    def time_of_remotion(self) -> float:
        return self._time_of_remotion
   

@dataclass(frozen=True,unsafe_hash=True)
class StlTask :
    """
    Signal Temporal Logic Task container class. This class is applied to store information about a given STL task like the predicate 
    the time interval of the task and the temporal operator
    
    Args:
        temporal_operator (Operator) : the temporal operator of the task
        predicate (PredicateFunction) : the predicate function of the task
        name
        
    """  
         
    predicate              : PredicateFunction 
    temporal_operator      : TemporalOperator  
    name                   : str              = None    
    
    @property
    def predicate_function(self):
        return self.predicate.function
    
    @property
    def contributing_agents(self):
        return self.predicate.contributing_agents
    
        
def go_to_goal_predicate_2d(goal:np.ndarray,epsilon :float, model_agent:DynamicalModel) ->PredicateFunction:
    
    try :
        position1 = model_agent.substates_dict[StateName.POSITION2D] #extract the required 2d position
    except :
        raise ValueError("The given dynamical models do not have a 2D position as a substate. The epsilon closeness predicate can only be applied to models with a 2D position as a substate")
    
    
    if position1.numel() != goal.size:
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(position1.numel()) + " and " + str(goal.size) + "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    
    if len(goal.shape) <2 :
        goal = goal[:,np.newaxis]
        
        
    predicate_expression = ( epsilon**2 - ca.dot((position1 - goal),(position1-goal)) ) # the scaling will make the time derivative smaller whe you built the barrier function
    predicate            = ca.Function("position_epsilon_closeness",[model_agent.state_vector],
                                                                    [predicate_expression],
                                                                    ["state_"+str(model_agent.id)],
                                                                    ["value"]) # this defined an hyperplane function

    return PredicateFunction(function=predicate)




def epsilon_position_closeness_predicate(epsilon:float, model_agent_i:DynamicalModel,model_agent_j:DynamicalModel) ->PredicateFunction:
    """
    Helper function to create a closeness relation predicate in the form ||position1-position2|| <= epsilon.
    This predicate is useful to dictate some closeness relation among two agents for example. The helper function can only be applied of 
    the models have a states StateName.POSITION2D as a substate. Otherwise the function will give an error.
    
    Args:
        epsilon  : the closeness value
        model_agent_i : the first agent
        model_agent_j : the second agent
    
    Returns:
        PredicateFunction : the predicate function 
        
    
    Example:
    >>> car_1 = DifferentialDrive(...)
    >>> car_2 = DifferentialDrive(...)
    >>> epsilon = 0.1
    >>> closeness_predicate = epsilon_position_closeness_predicate(epsilon,car_1,car_2)
    
    """

    try :
        position1 = model_agent_i.position #extract the required 2d position
        position2 = model_agent_j.position 
    except :
        raise ValueError("The given dynamical models do not have a 2D position as a substate. The epsilon closeness predicate can only be applied to models with a 2D position as a substate")
    

    if position1.shape != position2.shape :
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(position1.shape) + " and " + str(position2.shape) + "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    
    predicate_expression =  ( epsilon**2 - ca.sumsqr(position1 - position2) ) # the scaling will make the time derivative smaller whe you built the barrier function
    predicate            = ca.Function("position_epsilon_closeness",[model_agent_i.state_vector,model_agent_j.state_vector],
                                                                    [predicate_expression],
                                                                    ["state_"+str(model_agent_i.id),"state_"+str(model_agent_j.id)],
                                                                    ["value"]) # this defined an hyperplane function

    return PredicateFunction(function=predicate)





def formation_predicate(epsilon:float, model_agent_i:DynamicalModel,model_agent_j:DynamicalModel,relative_pos:np.ndarray,direction_i_to_j:bool=True) ->PredicateFunction:
    """
    Helper function to create a closeness relation predicate witha  certain relative position vector. in the form ||position1-position2|| <= epsilon.
    This predicate is useful to dictate some closeness relation among two agents for example. The helper function can only be applied of 
    the models have a states StateName.POSITION2D as a substate. Otherwise the function will give an error.
    
    Args:
        epsilon  : the closeness value
        model_agent_i : the first agent
        model_agent_j : the second agent
    
    Returns:
        PredicateFunction : the predicate function 
        
    
    Example:
    >>> car_1 = DifferentialDrive(...)
    >>> car_2 = DifferentialDrive(...)
    >>> epsilon = 0.1
    >>> closeness_predicate = epsilon_position_closeness_predicate(epsilon,car_1,car_2)
    
    """

    try :
        position_i = model_agent_i.position #extract the required 2d position
        position_j = model_agent_j.position 
    except :
        raise ValueError("The given dynamical models do not have a 2D position as a substate. The epsilon closeness predicate can only be applied to models with a 2D position as a substate")
    

    if position_i.shape != position_j.shape :
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(position_i.shape) + " and " + str(position_j.shape) + "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    if relative_pos.shape[0] != position_i.shape[0] :
        raise ValueError("The relative position vector must have the same dimension as the position of the agents. Agents have position dimension " + str(position_i.shape) + " and the relative position vector has dimension " + str(relative_pos.shape) )
    
    if direction_i_to_j :
        predicate_expression =  ( epsilon**2 - (position_j - position_i- relative_pos).T@(position_j-position_i - relative_pos) ) # the scaling will make the time derivative smaller whe you built the barrier function
    else :
        predicate_expression =  ( epsilon**2 - (position_i - position_j- relative_pos).T@(position_i-position_j - relative_pos) )
    
    predicate            = ca.Function("position_epsilon_closeness",[model_agent_i.state_vector,model_agent_j.state_vector],
                                                                    [predicate_expression],
                                                                    ["state_"+str(model_agent_i.id),"state_"+str(model_agent_j.id)],
                                                                    ["value"]) # this defined an hyperplane function

    return PredicateFunction(function=predicate)





def collision_avoidance_predicate(epsilon:float, model_agent_i:DynamicalModel,model_agent_j:DynamicalModel) :


    position1 = model_agent_i.substates_dict[StateName.POSITION2D] #extract the required 2d position
    position2 = model_agent_j.substates_dict[StateName.POSITION2D] 
    

    if position1.shape != position2.shape :
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(position1.shape) + " and " + str(position2.shape) + "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")

    predicate_expression =  (position1 - position2).T@(position1-position2) -  epsilon**2 

    predicate            = ca.Function("collision_avoidance",[model_agent_i.state_vector,model_agent_i.state_vector],
                                                             [predicate_expression],["state_"+str(model_agent_i.ID),"state_"+str(model_agent_j.ID)],
                                                             ["value"]) # this defined an hyperplane function


    return PredicateFunction(function=predicate)



def compute_ellipse_matrix(semi_major_axis: float = 1, semi_minor_axis: float = 0.5, theta: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the P and B matrix of an ellipse given its parameters.
    
    Args:
        semi_major_axis (float): The semi-major axis of the ellipse in the x-direction.
        semi_minor_axis (float): The semi-minor axis of the ellipse in the y-direction.
        theta (float): The rotation angle of the ellipse in radians (positive counter-clockwise direction).
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the P matrix and the B matrix.
            - P: A matrix representing the ellipse in matrix form (x-center)'P(x-center) <= 1.
            - B: A matrix representing the transformation from the unit circle to the ellipse Bx + center with ||x|| <= 1.
    
    Raises:
        Exception: If the axes ratio is not greater than 1.
    """
    
    if semi_major_axis / semi_minor_axis <= 1:
        raise Exception("Axes ratio must be greater than 1")
    
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    P = rotation_matrix.T @ np.diag([1 / semi_major_axis**2, 1 / semi_minor_axis**2]) @ rotation_matrix
    B = rotation_matrix @ np.diag([semi_major_axis, semi_minor_axis])
    return P, B



def relative_position_ellipsoidal_formation_predicate(P: np.ndarray, relative_formation: np.ndarray, source_model: DynamicalModel, target_model: DynamicalModel) ->PredicateFunction:
    """
    Constructs a predicate function for relative position ellipsoidal formation.

        Args:
            P (np.ndarray): The ellipsoidal matrix.
            relative_formation (np.ndarray): The relative formation vector.
            source_model (DynamicalModel): The source dynamical model.
            target_model (DynamicalModel): The target dynamical model.

        Returns:
            Callable[[np.ndarray], bool]: The constructed predicate function.

        Raises:
            ValueError: If the position dimensions of the two dynamical models are different.
            ValueError: If the matrix P is not symmetric.
            ValueError: If the dimension of relative_formation is inappropriate.
    """
    position1 = source_model.position #extract the required 2d position
    position2 = target_model.position 
    
    if position1.shape != position2.shape :
        raise ValueError("The two dynamical models have different position dimensions. Namely " + str(position1.shape) + " and " + str(position2.shape) + "\n If you want to construct an epsilon closeness predicate use two models that have the same position dimension")
    
    edge = position2 - position1
    n, m = np.shape(P)
    if n != m:
        raise ValueError("matrix should be symmetric")
    if len(relative_formation) != n:
        raise ValueError("variable 'center' has inappropriate dimension : matrix is " + str(n) + "x" + str(n) + ", while vector is dimension" + str(len(relative_formation)))

    if relative_formation.ndim == 1:
        relative_formation = relative_formation[:, np.newaxis] # put as column

    predicate_expression = (edge - relative_formation).T @ P @ (edge - relative_formation) - 1   # retun the predicate . # in this case the vector e is the edge ij

    predicate = ca.Function("collision_avoidance",[source_model.state_vector,source_model.state_vector],
                                                             [predicate_expression],["state_"+str(source_model.ID),"state_"+str(target_model.ID)],
                                                             ["value"]) # this defined an hyperplane function

    return PredicateFunction( function=predicate)



def expression_to_function_wrapper(expression: ca.MX, list_of_models: tuple[DynamicalModel],time_variable:ca.MX=None):
    """
    Wraps the given predicate expression with the input arguments.

    Args:
        predicate_expression (ca.MX)                : The predicate expression to be wrapped.
        list_of_models       (tuple[DynamicalModel]): Variable number of models objects. The order matters as the input list will be given in the order in which the agents are given

    Returns:
        ca.Function: A function object representing the wrapped predicate expression.
    """

    input_names = []
    inputs = []
    for model in list_of_models:
        if not isinstance(model, DynamicalModel):
            raise TypeError("All the initial arguments must be Agent objects. The last argument will be the predicate expression.")
        else:
            input_names.append("state_" + str(model.id))
            inputs += [model.state_vector]
    
    if time_variable != None :
        input_names.append("time")
        inputs += [time_variable]
    
    try :
        return ca.Function("costume_predicate", inputs, [expression], input_names, ["value"])
    except Exception as e:
        print(e)
        raise ValueError("The given expression is not a valid expression. The expression need to be a function of the states of the models given as input and possibly of time (if this was given as input.",
                         "Common problems are :1) the expression does not depend from the some or all the models (and thus their states) 2) the expression containrs the time but yout did not ptovide the time variable")
    



def conjunction_of_barriers(*args:BarrierFunction,associated_alpha_function:ca.Function=None)-> BarrierFunction :
    """
    Function to compute the conjunction of barrier functions. The function takes a variable number of barrier functions as input and returns a new barrier function that is the conjunction of the input barrier functions.
    
    Args:
        *args (BarrierFunction): Variable number of barrier functions.
    
    Returns:
        BarrierFunction: The barrier function representing the conjunction of the input barrier functions.
        
    Example :
    >>> b1 = BarrierFunction(...)
    >>> b2 = BarrierFunction(...)
    >>> b3 = BarrierFunction(...)
    >>> b4 = conjunction_of_barriers(b1,b2,b3)
    """
    
    # check if the input is a list of barrier functions
    for arg in args:
        if not isinstance(arg, BarrierFunction):
            raise TypeError("All the input arguments must be BarrierFunction objects.")
    
    # check that function is a scalar function
    if associated_alpha_function != None :
        if not isinstance(associated_alpha_function,ca.Function) :
            raise ValueError("The associated alpha function must be a casadi function")
        if associated_alpha_function.n_in() != 1 :
            raise ValueError("The associated alpha function must be a scalar function of one variable")
        if not  associated_alpha_function.size1_in(0) == 1 :
            raise ValueError("The associated alpha function must be a scalar function of one variable")
        
    
    contributing_agents = set()
    for barrier in args:
        contributing_agents.update(barrier.contributing_agents)
        
    # the barriers are all functions of some agents state and time. So we create such variables   minimum_approximation -> -1/eta log(sum -eta * barrier_i)
    inputs = {}
    
    for agent_id in contributing_agents :
        inputs["state_"+str(agent_id)] = ca.MX.sym("state_"+str(agent_id),barrier.function.size1_in("state_"+str(agent_id)))
    
    inputs["time"] = ca.MX.sym("time",1)
    dummy = ca.MX.sym("dummy",1)
    
    # each barrier function has an associates switch function. This function is equal to 1 if the barrier is active and 0 otherwise
    # we then use this function to create another switch. This switch is used to remove the barrier from the minimum
    # create at the conjunction. Namely, the switch sets the barrier to infinity when the barrier is not needed anymore.
    
    inf_switch = ca.Function("inf_switch",[dummy],[ca.if_else(dummy==1.,1.,10**20)]) 
    sum_switch = 0 # sum of the switches will be the final switch. namely, it will be zero when all the switches are zero
    
    
    min_list = []
    sum = 0
    eta = 10
    
    
    for barrier in args :
        # gather the inputs for this barrier
        barrier_inputs = {}
        switch : ca.Function = barrier.switch_function
        
        for agent_id in barrier.contributing_agents :
            barrier_inputs["state_"+str(agent_id)] = inputs["state_"+str(agent_id)] # gather the inputs for this barrier
        barrier_inputs["time"] = inputs["time"] 
        sum_switch += switch(dummy) 
        
        
        sum += ca.exp(-eta*barrier.function.call(barrier_inputs)["value"]) # sum of the exponentials
        
        min_list.append(  barrier.function.call(barrier_inputs)["value"] + inf_switch(switch(barrier_inputs["time"]))  )
    
    
    #create the final switch function
    final_switch = ca.Function("final_switch",[dummy],[ca.if_else(sum_switch>=1,1,0)]) # the final switch is the switch that is zero when all the switches are zero
    
    # smooth min
    # conjunction_barrier = -1/eta*ca.log(sum) # the smooth minimum of the barriers
    # real min
    conjunction_barrier = ca.mmin(ca.vertcat(*min_list)) # the barrier function is the minimum of the barriers
    
    b = ca.Function("conjunction_barrier",list(inputs.values()),[conjunction_barrier],list(inputs.keys()),["value"]) # Now we can have conjunctions of formulas
    
    return BarrierFunction(function=b,associated_alpha_function=associated_alpha_function,switch_function=final_switch)


def create_barrier_from_task(task:StlTask,initial_conditions :dict[UniqueIdentifier,np.ndarray],alpha_function: ca.Function = None,t_init:float = 0 ) -> BarrierFunction:
    """
    Creates a barrier function from a given STLtask in the form of b(x,t) = mu(x) + gamma(t-t_init) 
    where mu(x) is the predicate and gamma(t) is a suitably defined time function 
    
    Args:
        task (StlTask)                            : the task for which the barrier function is to be created
        initial_conditions (dict[UniqueIdentifier,np.ndarray]) : the initial conditions of the agents
        alpha_function (ca.Function)              : the associated alpha function to be used for barrier constraint construction
        t_init (float)                            : the initial time of the barrier function in terms of the time to which the barrier should start
        
    return :
        BarrierFunction : the barrier function associated with the given task
    
    """
    # get task specifics
    contributing_agents  = task.contributing_agents # list of contributing agents. In the case of this controller this is always going to be 2 agents : the self agent and another agent
    
    # check that all the agents are present
    if not all([agent_id in initial_conditions.keys() for agent_id in contributing_agents]) :
        raise ValueError("The initial conditions for the contributing agents are not complete. Contributing agents are " + str(contributing_agents) + " and the initial conditions are given for " + str(initial_conditions.keys()))
    
    # check that the sates sizes match
    for agent_id in contributing_agents :
        if not (task.predicate.function.size1_in("state_"+str(agent_id)) == initial_conditions[agent_id].shape[0]) :
            raise ValueError("The initial condition for agent " + str(agent_id) + " has a different size than the state of the agent. The size of the state is " + str(task.predicate.function.size1_in("state_"+str(agent_id))) + " and the size of the initial condition is " + str(initial_conditions[agent_id].shape[0]))
        
    
    # determine the initial values of the barrier function 
    initial_inputs   = {state_name_str(agent_id) : initial_conditions[agent_id] for agent_id in contributing_agents} # create a dictionary with the initial state of the agents
    symbolic_inputs = {}
    
    # STATE SYMBOLIC INPUT
    for agent_id in contributing_agents:
        symbolic_inputs["state_"+str(agent_id)] = ca.MX.sym("state_"+str(agent_id),task.predicate.function.size_in("state_"+str(agent_id))) # create a dictionary with the initial state of the agents
    
    
    predicate_fun = task.predicate.function
    predicate_initial_value =  predicate_fun.call(initial_inputs)["value"]
    symbolic_predicate_value = predicate_fun.call(symbolic_inputs)["value"]
        
    
    # gamma(t) function construction :
    # for always, eventually and Eventually always a linear decay function is what we need to create the barrier function.
    # on the other hand we need to add a sinusoidal part to guarantee the task satisfaction. The following intuition should help :
    # 1) An EventuallyAlways just says that an always should occur within a time in the eventually. So it is just a kinf od delayed always where the delay it is the time of satisfaction within the eventually
    # 2) An AlwaysEventually is a patrolling task. So basically this task says that at every time of the always, a certain predicate should evetually happen. This is equivalent to satisify the eventual;ly at regular periods within the always
    
    # we first construct the linear part the gamma function
    
    
    time_of_satisfaction = task.temporal_operator.time_of_satisfaction
    time_of_remotion     = task.temporal_operator.time_of_remotion
    # for each barrier we now create a time transient function :
    time_var                  = ca.MX.sym("time",1) # create a symbolic variable for time 
    
    # now we adopt a scaling of the barrier so that the constraint is more feasible. 
    
    if (time_of_satisfaction-t_init) < 0 : # G_[0,b] case
        raise ValueError("The time of satisfaction of the task is less than the initial time of the barrier. This is not possible")
    
    if (time_of_satisfaction-t_init) == 0 : # G_[0,b] case
        gamma    = 0
    
    else :
        # normalization phase (this helps feasibility heuristically)
        
        normalization_scale = np.abs((time_of_satisfaction-t_init)/predicate_initial_value) # sets the value of the barrier equal to the tme at which I ned to satisfy the barrier. It basciaky sets the spatial and termpoal diemsnion to the same scale 
        predicate_fun = ca.Function("scaled_predicate",list(symbolic_inputs.values()),[task.predicate.function.call(symbolic_inputs)["value"]*normalization_scale],list(symbolic_inputs.keys()),["value"])
        predicate_initial_value =  predicate_fun.call(initial_inputs)["value"]
        symbolic_predicate_value = predicate_fun.call(symbolic_inputs)["value"]
        
        if (time_of_satisfaction-t_init) == 0 : # G_[0,b] case
            gamma    = 0
            
        else :  
            
            if predicate_initial_value <=0:
                gamma0 = - predicate_initial_value*1.2 # this gives you always a gamma function that is at 45 degrees decay. It is an heuristic
            if predicate_initial_value >0:
                gamma0 =  - predicate_initial_value*0.8
                
            a = gamma0/(time_of_satisfaction-t_init)**2
            b = -2*gamma0/(time_of_satisfaction-t_init)
            c = gamma0
            quadratic_decay = a*(time_var-t_init)**2 + b*(time_var-t_init) + c
            gamma    = ca.if_else(time_var <=time_of_satisfaction-t_init ,quadratic_decay,0) # piece wise linear function
            
            # gamma    = ca.if_else(time_var <=time_of_satisfaction-t_init ,gamma0 + (time_var - t_init)*slope,0) # piece wise linear function
            # # exponential version
            # gamma0   = - predicate_initial_value # this gives you always a gamma function that is at 45 degrees decay. It is an heuristic
            # tau      = -1/(time_of_satisfaction-t_init)*np.log(0.04/gamma0) # negative decxay rate of the barrier. The barrier will decay linearly from gamma0 to zero
            # gamma    = ca.if_else(time_var <=time_of_satisfaction,gamma0*ca.exp(-tau*(time_var-t_init)),0) # piece wise linear function
        

    
    if isinstance(task.temporal_operator, AlwaysEventuallyOperator):
        # we need to add a sinusoidal part to the gamma function
        frequency    = (np.pi/(task.temporal_operator.period)) # only pi and not 2 pi. Every pi the sin goes to zero to zero once only, which is what we need
        amplitude    = 10 # regulates the freedom basically. We need to find an heuristic for this term.
        gamma_sin    = ca.if_else(time_var >time_of_satisfaction,amplitude*ca.fabs(ca.sin(frequency*(time_var-time_of_satisfaction))),0) 
        # this sinusoidal component does the following :
        # 1) Starts oscillating from the time of satsifaction (start of the patrolling)
        # 2) The function is always positive and oscillates between 0 and amplitude (this is why the absolute value is taken)
        gamma = gamma + gamma_sin # add sinusoidal component to linear component
    
    switch_function = ca.Function("switch_function",[time_var],[ca.if_else(time_var<= time_of_remotion,1.,0.)]) # create the gamma function
    gamma_fun       = ca.Function("gamma_function",[time_var],[gamma]) # create the gamma function
    
    
    
    # now add time to the inputs and find symbolic value of the barrier function
    symbolic_inputs["time"] = time_var # add time to the inputs
    barrier_symbolic_output  = (symbolic_predicate_value + gamma) 
    
    # now we create the barrier function. We will need to create the input/output names and the symbolic inputs
    barrier_fun = ca.Function("barrierFunction",list(symbolic_inputs.values()),[barrier_symbolic_output],list(symbolic_inputs.keys()),["value"])
    
    return BarrierFunction(function = barrier_fun, 
                           associated_alpha_function = alpha_function,
                           time_function=gamma_fun,
                           switch_function=switch_function)
    
   