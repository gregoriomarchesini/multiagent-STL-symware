# Dynamical models for multi-agent simulation

import casadi as ca
import numpy as np
from abc import ABC, abstractmethod
from enum import StrEnum
from itertools import product
from typing import TypeAlias
from symaware.base import Identifier
from kth.utils import is_casadiMX,wrap_name,get_id_from_name,name_without_id


# The enumerators are used to define some input and output names that can be useful 
# for describing a dynamical system. Feel free to add more names if you need them


class InputAffineDynamicalSymbolicModel(ABC):
    """
    Abstract class to store the dynamical model of a given system.
     .
     x = f(x) + g(x)u
    """
    def __init__(self,ID: Identifier , discretization_time_step:float) -> None:
        self._ID = ID # IDof the agent that the dynamical system refers to
        self._discretization_time_step = discretization_time_step # the time step used to discretize the system
    
    @property
    def ID(self) -> Identifier:
        return self._ID
    
    @property
    def discretization_time_step(self) -> float:
        return self._discretization_time_step
    
    @property
    @abstractmethod
    def state_vector_sym(self) -> ca.MX:
        """
        Define a state vector for your system. For example you could have the following state vector for a car
        >>> self._position = ca.MX.sym(wrap_name(StateName.POSITION2D,self._ID), 2) # create a casadi variable with the name "position_2d_id" where IDof the agent (this is useful for debuggin so that states from different dynamical systems have different names)
        >>> self._heading  = ca.MX.sym(wrap_name(StateName.HEADING,self._ID))       # heading state
        """
        return ca.MX.sym("state_vector",0,0)
    
    @property
    @abstractmethod
    def control_input_sym(self) -> ca.MX:
        """
        Define an input vector for your system. For example you could have the following input vector for a car
        >>> self._velocity     = ca.MX.sym(wrap_name(InputName.VELOCITY1D,self._ID)) # longitudinal velocity
        >>> self._turning_rate = ca.MX.sym(wrap_name(InputName.TURNING_RATE,self._ID)) # tunring rate
        >>> self._input_vector = ca.vertcat(self._velocity,self._turning_rate)
        """
        
        return  ca.MX.sym("input_vector",0,0)
    
 
    @property
    @abstractmethod
    def input_constraints_A(self)->ca.MX:
        """
        Matrix of the input constraints A@input_vector<=b
        
        Note : Nonlinear constraints should be approximated to be linear. A Polygon with sufficiently many faces is sufficient to represent closed convex sets.
        Support function to construct such polygons are provided
        """
        pass
    
    
    @property
    @abstractmethod
    def input_constraints_b(self)->ca.MX:
        """
        Vector of the input constraints A@input_vector<=b
        
        Note : Nonlinear constraints should be approximated to be linear. A Polygon with sufficiently many faces is sufficient to represent closed convex sets.
        Support function to construct such polygons are provided
        """
        pass
    
    @property
    @abstractmethod
    def dynamics_sym(self) -> ca.MX:
        """casadi expression for the dynamical model of the system as a function of the symbolic input and the symbolic state"""
        pass
        
    
    @property
    def dynamics_fun(self)->ca.Function:
        return ca.Function('concrete_dynamics',[self.state_vector_sym,self.control_input_sym],[self.dynamics_sym],["state","input"],["value"])
    
    def lie_derivative_g(self,b:ca.MX) -> ca.MX: 
        """
        return the lie derivative of b along the system(expression in casadi MX)
        Note that the expression should be a function of the state_vector that is an attribute of the dynamical model 
        """
        return ca.jacobian(b,self.state_vector_sym) @ self.g 
    
    def lie_derivative_f(self,b:ca.MX) -> ca.MX:
        """
        return the lie derivative of b along the system (expression in casadi MX)
        Note that the expression is a function should be a function the state_vector that is an attribute of the dynamical model 
        """
        return ca.jacobian(b,self.state_vector_sym) @ self.f
    
    def lie_derivative_f_function(self, b:ca.Function) -> ca.Function:
        """
        Given a barrier function in the form b("state_1","state_2",...."time") is returns the Lie derivative along f as of the dynamical system
        as a function of the states and time. The state of the current dynamical system should also be among the states or an error will be thrown
        
        Args :
            b : The barrier function expressed as a casadi function
        
        Returns :
            A casadi function that takes the states and time as input and returns the Lie derivative of b along f
            
        """
        
        input_names = b.name_in()
        ids  = [get_id_from_name(name) for name in input_names if "state" in name.split('_')]
        if not self._ID in ids:
            raise ValueError(f"The barrier function {b} does not contain the state of the current dynamical system. states found are {ids} but agent has state {self._ID}")
        
        
        named_inputs = {} # symbolic inputs to be given to the function
        for name in input_names:
            if name == "time":
                named_inputs[name] = ca.MX.sym(name)
            else :
                ID= get_id_from_name(name)
                
                if ID == None:
                    raise ValueError(f"The barrier function {b} has an input called {name} that is not among the states of the dynamical system. This is not allowed. Only names like 'state_id' and 'time' are allowed")
                
                if ID== self._ID:
                    named_inputs[name] = self.state_vector_sym
                else: 
                    named_inputs[name] = ca.MX.sym(name,b.size1_in(name))
            
        b_exp = b.call(named_inputs)["value"]
        lie_f_b = ca.jacobian(b_exp,self.state_vector_sym) @ self.f
        
        
        return ca.Function("lie_f_"+b.name(),list(named_inputs.values()),[lie_f_b],list(named_inputs.keys()),["value"]) 
        
    def lie_derivative_g_function(self, b:ca.Function) -> ca.Function:
        """
        Given a barrier function in the form b("state_1","state_2",...."time") is returns the Lie derivative along g as of the dynamical system
        as a function of the states and time. The state of the current dynamical system should also be among the states or an error will be thrown
        
        Args :
            b : The barrier function expressed as a casadi function
        Returns :
            A casadi function that takes the states and time as input and returns the Lie derivative of b along g
        
        Example :
        >>> state = DynamicalModel.state_vector
        >>> b_exp = state.T@state 
        >>> b = ca.Function("b",[state],[b_exp],["state_1","time"],["value"])
        >>> b = ca.Function("b",["state_1","state_2","time"],["state_1**2+state_2**2"])
        >>> lie_g_b = self.lie_derivative_g_function(b)
            
        """
        
        input_names = b.name_in()
        ids  = [get_id_from_name(name) for name in input_names if "state" in name.split('_')]
        if not self._ID in ids:
            raise ValueError(f"The barrier function {b} does not contain the state of the current dynamical system. states found are {ids} but agent has state {self._ID}")
        
        
        named_inputs = {} # symbolic inputs to be given to the function
        for name in input_names:
            if name == "time":
                named_inputs[name] = ca.MX.sym(name)
            else :
                ID= get_id_from_name(name)
                if ID== self._ID:
                    named_inputs[name] = self.state_vector_sym
                else: 
                    named_inputs[name] = ca.MX.sym(name,b.size1_in(name))
            
        b_exp = b.call(named_inputs)["value"]
        lie_g_b = ca.jacobian(b_exp,self.state_vector_sym) @ self.g
        
        
        return ca.Function("lie_f_"+b.name(),list(named_inputs.values()),[lie_g_b],list(named_inputs.keys()),["value"])    
    
    
    def is_relative_degree_1(self, b:ca.MX) -> int:    
        directional_gradient = self.lie_derivative_g(b)
        return directional_gradient.is_zero()

    
    
    
class SingleIntegrator( InputAffineDynamicalSymbolicModel):
   
        
    def __init__(self, max_velocity: float,ID:Identifier) -> None:
        
        super().__init__(ID=ID)
        
        self._max_velocity = max_velocity
        if self._max_velocity <= 0:
            raise ValueError("Max velocity must be positive.")

        # create the state variables
        self._position = ca.MX.sym("position_"+str(self.ID), 2)
        self._velocity = ca.MX.sym("velocity_"+str(self.ID), 2)

        # create dynamics model
        self._g = np.eye(2)
        self._f = ca.vertcat(0,0)
        

        self._input_vector = self._velocity
        self._state_vector = self._position
        self._input_constraints_A, self._input_constraints_b,self._input_constraints_vertices = create_approximate_ball_constraints2d(radius=self._max_velocity,points_number=40)
    
    @property
    def max_velocity(self) -> float:
        return self._max_velocity

    @property
    def g(self)->ca.MX:
        return self._g
    @property
    def f(self)->ca.MX:
        return self._f
   
    @property
    def input_vector_sym(self) -> ca.MX:
        return self._input_vector

    @property
    def state_vector_sym(self) -> ca.MX:
        return self._state_vector

    @property
    def dynamics_sym(self) -> ca.MX:
        return self._f + self._g @ self._input_vector

    
    @property
    def input_constraints_A(self)->ca.MX:
        """Matrix of the input constraints Au<=b"""
        return self._input_constraints_A
    
    @property
    def input_constraints_b(self)->ca.MX:
        """Vector of the input constraints Au<=b"""
        return self._input_constraints_b
    
    @property
    def position(self) -> ca.MX:
        return self._position
    
    @property
    def maximum_expressible_speed(self)->float:
        return self._max_velocity
    
        

    
    


class DifferentialDrive( InputAffineDynamicalSymbolicModel):
    """
    Class Differential Drive - A simple model of a differential drive robot.
    
    Args:
        wheel_base (float): The distance between the front and rear axles of the car.
        max_velocity (float): The maximum velocity capability of the car.
        angular_velocity (float): The maximum steering angle of the car.
    """
    count = 0
     
    def __init__(self, max_speed: float, max_angular_velocity: float, look_ahead_distance: float,ID: Identifier) -> None:
        
        
        super().__init__(ID=ID)

    
        self._max_speed             = max_speed            # max longitudinal speed
        self._max_angular_velocity  = max_angular_velocity # max angular velocity
        self._look_ahead_distance   = look_ahead_distance  # diffeormorphism to have a first order model of speed which respect to the angular vecolocity

        if self._max_speed <= 0:
            raise ValueError("Max velocity must be positive.")
        if  self._max_angular_velocity  <= 0:
            raise ValueError("Max angular velocity must be positive.")
        

        # create the state variables. Node that a unique IDis given for debugging purposes (KEEP THIS TEMPLATE)
        self._position = ca.MX.sym("position_"+str(ID), 2)
        self._heading  = ca.MX.sym("heading_"+str(ID))

        # create dynamics model
        row1 = ca.horzcat(ca.cos(self._heading), -self._look_ahead_distance*ca.sin(self._heading))
        row2 = ca.horzcat(ca.sin(self._heading), self._look_ahead_distance*ca.cos(self._heading))
        row3 = ca.horzcat(0,1)
        self._g = ca.vertcat(row1,row2,row3)
        self._f = ca.vertcat(0,0,0)

        # control input
        self._turning_rate = ca.MX.sym("turning_rate_"+str(ID)) # tunring rate
        self._velocity = ca.MX.sym("velocity_"+str(ID)) # longintudinal velocity

        #input vector
        self._input_vector = ca.vertcat(self._velocity,self._turning_rate)
        #state vector
        self._state_vector = ca.vertcat(self._position,self._heading)
        #the order must respect the one given in the state and input vector!
        self._input_constraints_A, self._input_constraints_b,self._input_constraints_vertices = create_box_constraint_function([[-self._max_speed, self._max_speed], [-self._max_angular_velocity, self._max_angular_velocity]])    
    
    @property
    def max_speed(self) -> float:
        return self._max_speed

    @property
    def max_angular_velocity(self) -> float:
        return self._max_angular_velocity

    @property
    def look_ahead_distance(self) -> float:
        return self._look_ahead_distance

    @property
    def g(self)->ca.MX:
        return self._g
    
    @property
    def f(self)->ca.MX:
        return self._f
    
    # define here the input vector and the state vector
    @property
    def input_vector(self) -> ca.MX:
        return self._input_vector

    @property
    def state_vector(self) -> ca.MX:
        return self._state_vector

    @property
    def dynamics_exp(self) -> ca.MX:
        return self._f + self._g @ self._input_vector
    

    @property
    def maximum_expressible_speed(self)->float:
        return np.sqrt(self._max_speed**2 + (self._max_angular_velocity*self._look_ahead_distance)**2)
    
    @property
    def input_constraints_A(self)->ca.MX:
        """Matrix of the input constraints Au<=b"""
        return self._input_constraints_A
    
    @property
    def input_constraints_b(self)->ca.MX:
        """Vector of the input constraints Au<=b"""
        return self._input_constraints_b





























# here we provide some support classes to create circular and box type constraints matriced Ax-b <=0 

def create_approximate_ball_constraints2d(radius:float,points_number:int)-> tuple[np.ndarray,np.ndarray,np.ndarray] :
    """
    Computes constraints matrix and vector A,b for an approximation of a ball constraint with a given radius.
    The how many planes will be used to approximate the ball. We cap the value of the planes to 80 to avoid numerical errors in the computation of the normals
    
    Args :
        radius : The radius of the ball
        points_number : The number of planes used to approximate the ball
        
    Returns:
        A : The constraints matrix and vector for the ball constraint
        b : The constraints matrix and vector for the ball constraint
        vertices : The vertices of the polygon that approximates the ball
        
    
    Notes :
    To compute a center shift in the constraints it is sufficient to recompute b as b = b -+ A@center_shift. Indeeed A(x-c)-b = Ax - Ac - b = Ax - (b + Ac) 
    To compute a scaled version of the obtained polygone it is sufficient to scale b  by a scalar gamma> 0. Ax-gamma*b<=0 is the polygon scaled by gamma
    
    """

    # create constraints
    Arows    = []
    brows    = []
    vertices = []
    
    points_number = min(points_number,80) # to avoid numerical error on the computation of the normals
    
    for i in range(0,points_number):
        angle = 2 * np.pi * i / (points_number-1)
        vertices += [np.array([radius*np.cos(angle), radius*np.sin(angle)])]
        
    for j in range(0,len(vertices)) :
        
        tangent = vertices[j] - vertices[j-1]
        norm_tangent = np.linalg.norm(tangent)
        outer_normal = np.array([tangent[1],-tangent[0]])/norm_tangent
        
        b = np.sqrt(radius**2 - (norm_tangent/2)**2)
        Arows += [outer_normal]
        brows += [np.array([[b]])]
    
    A = np.row_stack(Arows)
    b = np.row_stack(brows)
    vertices = np.row_stack(vertices)

    return A,b,vertices


def create_box_constraint_function(bounds: list[list[float,float]])-> tuple[np.ndarray,np.ndarray,np.ndarray] :
    """
    Create a casadi function that checks if a given vector is inside the box bounds.
    
    Args :
        bounds : The box bounds in the form np.array([[min1, max1], [min2, max2], ..., [minN, maxN]])
    
    Returns:
        A : The constraints matrix and vector for the ball constraint
        b : The constraints matrix and vector for the ball constraint
        vertices : The vertices of the polygon that approximates the ball
        
    Notes :
    To compute a center shift in the constraints it is sufficient to recompute b as b = b -+ A@center_shift. Indeeed A(x-c)-b = Ax - Ac - b = Ax - (b + Ac) 
    To compute a scaled version of the obtained polygone it is sufficient to scale b  by a scalar gamma> 0. Ax-gamma*b<=0 is the polygon scaled by gamma
    """

    base  = np.eye(len(bounds))
    Arows = []
    brows = []
    vertices = np.array(list(product(*bounds)))

    for i in range(len(bounds)):
        Arows.append(base[i, :])
        Arows.append(-base[i, :])
        try :
            brows.append(bounds[i][1])
            brows.append(-bounds[i][0])
        except :
            raise ValueError(f"Bounds should be a list of lists with two elements. Got {bounds} instead")

    A = np.row_stack(Arows)
    b = np.row_stack(brows)
    vertices = np.row_stack(vertices)

    return A,b,vertices

