from   symaware.base.data import Identifier
import casadi as ca
import numpy as np
from   typing import TypeAlias
import casadi.tools as casadi_tools
from   collections import namedtuple

from   kth.signal_temporal_logic import BarrierFunction
from   kth.utils import NoStdStreams, MathematicalDynamicModel


XGuess: TypeAlias = casadi_tools.structure3.msymStruct  # type alias
PGuess :TypeAlias = casadi_tools.structure3.msymStruct  # type alias
Sample = namedtuple("sample", "x_guess parameters_value")


class LipschitzSolver:
    """
    This is a parent class for solving the adaptive Lipschitz computation. It is a parent class for the NuSolver and UpsilonSolver classes.
    The class implements the methods for solving the optimization problem for the adaptive Lipschitz constant.
    
    """
    def __init__(self,barrier:BarrierFunction) -> None:
        self.solver = None
        self.barrier= barrier 
        
    def _solve_lipschitz(self,sample:Sample) -> float:
        """
        Fast parametric program applied to to find the optimal upsilion_value.
        A sample contains a starting guess for the optimization and a set of paparemeters. The parameters are mainly the 
        reachable set of the agents involved in the computation of the Lipschitz constant while the initial guess is a randomly generated set of initial conditions

        Args:
            initial_guess_sample (casadi_tools.structure3.msymStruct): inital guess for the optimization problem
            parameters_sample (casadi_tools.structure3.msymStruct)   : set of aparameters to be used for the optimization

        Returns:
            result of the optimization
        """
        
        res = self.solver(ca.vertcat(sample.x_guess),ca.vertcat(sample.parameters_value))
       
        return float(res)
    
    
    def solve(self,samples: list[Sample],parallelize : bool= False) -> float:
        """
        Computes the optimal upsilon value based on an a list of samples. The minimum value of the 
        Lipschitz constant is taken as the correct one.

        Args:
            current_time (float): The current time.
            state_agent_i (np.ndarray): The state of agent i.
            state_agent_j (np.ndarray, optional): The state of agent j. Defaults to None.
            num_samples (int, optional): The number of samples to generate. Defaults to 10.

        Returns:
            optimal_value (float): The optimal upsilon value.
        """
        
        
        #  THIS IS THE PARALLEL VERSION BUT IT DOEN"T WORK BECAUSE CASADI CANNOT BE PICKLED. PROBLEM TO BE RESOLVED IN THE FUTURE
        if  parallelize : # just because parallelising with few iterations is inefficient (we should do a test to see where the efficieny point breaks)
            raise NotImplementedError("Parallelization is not implemented yet")
            mp_pool = mp.Pool(processes=6)
            results = mp_pool.map(self._solve_lipschitz,samples)
            
            minimum_value = np.min(results)
            return minimum_value 
        
        # THIS IS EFFECTIVELY THE ONLY OPTION FOR NOW
        else :
            values = []
            for sample in samples:
                try :
                    values.append(self._solve_lipschitz(sample))
                except Exception as e:
                    print(e)
                    continue
                    
            
            if len(values) == 0:
                raise RuntimeError("All the samples failed to solve the optimization problem. Exciting... ")
            
            minimum_value = np.min(values)
            return minimum_value



class NuSolver(LipschitzSolver):
    """
    Solver for the lipschitz constant for single agent tasks. In this case the Lipschitz constant of the barrier depend on the state of a single agent and its reachable set.
    """
    def __init__(self,barrier:BarrierFunction,model_agent_i:MathematicalDynamicModel) -> None:
        """
        
        Initialise the solver
        Args:
            barrier (BarrierFunction): barrier function for which it is desired to find a Lipschitz constant
            model_agent_i (MathematicalDynamicModel): model of the agent from which the barrier depends

        Raises:
            Exception: If the barrier function depends on the state of more than one agent
            Exception: If the agent is not involved in the barrier function at all
            Exception: If the barrier function does not have an associated alpha function
        """
        
        
        super().__init__(barrier=barrier)
        
        self._warm_start_solution = []
        
        if len(self.barrier.contributing_agents) != 1:
            raise Exception("The given barrier function depends on mure than one agent. Nu can only be computed for a single agent barrier function. Two agents will instead use two independent nu solvers")
        
        if model_agent_i.unique_identifier not in self.barrier.contributing_agents:
            raise Exception("The given barrier function does not depend on the given agent")

        self.model_agent_i = model_agent_i
        
        entry_state_i = [casadi_tools.entry("state_"+str(self.model_agent_i.unique_identifier),shape=self.model_agent_i.state_vector.size1())]
        entry_time    = [casadi_tools.entry('time',shape=1)]
        entry_input   = [casadi_tools.entry('input',shape=self.model_agent_i.input_vector.size1())]
            
        self.parameters = casadi_tools.struct_symMX( entry_state_i + entry_time)
        self.x          = casadi_tools.struct_symMX( entry_state_i + entry_time+ entry_input)
        
        # compute Lie derivatives
        lie_fi_fun : ca.Function = self.model_agent_i.lie_derivative_f_function(self.barrier.function)
        lie_gi_fun : ca.Function = self.model_agent_i.lie_derivative_g_function(self.barrier.function) 
        barrier_time_derivative : ca.Function = self.barrier.partial_time_derivative
        
        if self.barrier.associated_alpha_function != None :
            alpha : ca.Function = self.barrier.associated_alpha_function
        else :
            raise Exception("The barrier function does not have an associated alpha function. Please provide one")
        
        
    
        # compute reachable sets (these will be parameteric as a function of the initial state)
        Ai,bi = self.model_agent_i.reachable_set_A_and_b(self.parameters["state_"+str(self.model_agent_i.unique_identifier)])

        # CREATE PARAMETRIC OPTIMIZATION PROGRAM
        
        # parameters of the optimization
        initial_parameters   = {"state_"+str(self.model_agent_i.unique_identifier):self.parameters["state_"+str(self.model_agent_i.unique_identifier)],
                                "time"                             :self.parameters["time"]}
        
        # optimization variables
        variables = {"state_"+str(self.model_agent_i.unique_identifier):self.x["state_"+str(self.model_agent_i.unique_identifier)],
                    "time"                        :self.x["time"]}
        
        # create the objective function 
        nlp_objective = -1* (           lie_fi_fun.call(initial_parameters)["value"]   -  lie_fi_fun.call(variables)["value"] + 
                                ( lie_gi_fun.call(initial_parameters)["value"]          -  lie_gi_fun.call(variables)["value"] )  @self.x["input"]  + 
                            (barrier_time_derivative.call(initial_parameters)["value"] -  barrier_time_derivative.call(variables)["value"]) +
                            (alpha(barrier.function.call(initial_parameters)["value"]) -  alpha(barrier.function.call(variables)["value"])) )

        
        
        # create the constraints
        state_i_constraint = Ai @ self.x["state_"+str(self.model_agent_i.unique_identifier)] - bi                              # reachable set constraints
        input_constraint   = self.model_agent_i.input_constraints_A @ self.x["input"] - self.model_agent_i.input_constraints_b # input constraints
        time_constraints   = ca.vertcat(self.x["time"] - self.parameters["time"] - (self.model_agent_i._time_step)
                                        ,  -((self.x["time"]) - self.parameters["time"] ))                                     # delta < time_step -> delta - time_step<0   AND delta >0 -> -delta < 0
        constraints        = ca.vertcat(state_i_constraint,input_constraint,time_constraints )                                 # stack all of the into a single vector
            
            
        
        opts = {
        'print_iteration': False,    # Suppress iteration output
        'print_header': False,       # Suppress solver header
        'print_status': False,       # Suppress solver status
        'print_time': False,         # Suppress timing information
        'verbose_init': 0,           # Set verbosity level to minimum
        'qpsol':'qpoases',
        'qpsol_options':{"printLevel":"none",'verbose': False,
                        "print_time":False,
                        "print_problem":False,}
        }
        
        
        with NoStdStreams(): # remove the output from the solver initialization
            nlp = {'x':self.x , 'f':nlp_objective, 'g':constraints, 'p':self.parameters}
            S = ca.nlpsol('S', 'sqpmethod', nlp,opts)
            self.solver = ca.Function('solver',[ca.vertcat(self.x),ca.vertcat(self.parameters)],[S(x0=self.x,p=self.parameters,ubg=0)['f']])
            
        
            
        
    def generate_samples(self,current_time: float, state_agent_i: np.ndarray, num_samples: int = 10) -> list[Sample]:
        """
        Generates samples for nu optimization. 
        
        Args:
            current_time (float)        : Current time from which the optimization is undertaken
            state_agent_i (np.ndarray)  : State of the agent for which the optimization is undertaken
            num_samples (int, optional) : Number of samples to be generated. Defaults to 10.

        Returns:
            list[Sample]: list of samples
        """
        

        samples = []
        
        # set paramseters
        p_value = self.parameters(0)
        p_value["state_" + str(self.model_agent_i.unique_identifier)] = state_agent_i
        p_value["time"] = current_time
        
        # random initial conditions are create by sampling 
        # the reachable set and the input constraint set for the agent.
        # To do that efficiently we use the fact that all these sets are polytopes which can be efficiently sampled using random convex combinations
        
        for jj in range(num_samples):
            x_guess = self.x(0)

            verticesi = self.model_agent_i.reachable_set_vertices(state_agent_i)
            input_vertices = self.model_agent_i._input_constraints_vertices

            num_vertices_i = verticesi.shape[0]
            num_vertices_input = input_vertices.shape[0]
            
            # sampling from a polygone using the random convex combinaitons
            convex_coeffi = np.random.rand(num_vertices_i)
            convex_coeffi = convex_coeffi / np.sum(convex_coeffi)
            convex_coeff_input = np.random.rand(num_vertices_input)
            convex_coeff_input = convex_coeff_input / np.sum(convex_coeff_input)
            

            
            time_sample = current_time + np.random.rand(1) * self.model_agent_i._time_step

            # random initial state 1
            statei_sample = convex_coeffi @ verticesi
            
            input_sample = convex_coeff_input @ input_vertices

            x_guess["state_" + str(self.model_agent_i.unique_identifier)] = statei_sample
            x_guess["time"]  = time_sample
            x_guess["input"] = input_sample
            
            samples.append(Sample(x_guess, p_value))

        
        return samples 
            
            
            

class  UpsilonSolver(LipschitzSolver):
    """
    Solver for the lipschitz constant for two agent tasks. In this case the Lipschitz constant of the barrier depend on the state of two agents and their reachable sets.
    The index "i" is the state of the first agent and the state of the second agent "j" is the state of the second agent.
    The agent "i" can compute i_upsilon_ij using this class. Another class needs to be defined to compute j_upsilon_ij by setting the agents in the opposite order.
    Note that nu_ij = i_upsilon_ij + j_upsilon_ji so that two classes are needed to compute the Lipschitz constant for the barrier function. Only half the Lipschitz constant is computed here

    """
    def __init__(self,barrier:BarrierFunction,model_agent_i:MathematicalDynamicModel,model_agent_j:MathematicalDynamicModel) -> None:
        """
        
        Initialise the solver
        Args:
            barrier (BarrierFunction): barrier function for which it is desired to find a Lipschitz constant
            model_agent_i (MathematicalDynamicModel): model of the first agent from which the barrier depends
            model_agent_j (MathematicalDynamicModel): model of the second agent from which the barrier depends.

        Raises:
            Exception: if the barrier function does not depend on the given agents
            Exception: if the barrier function does not have an associated alpha function
        """
        
        super().__init__(barrier=barrier)
        
        if (model_agent_i.unique_identifier not in self.barrier.contributing_agents) or (model_agent_j.unique_identifier not in self.barrier.contributing_agents) :
            raise Exception(f"The given barrier function does not depend on the given agents. Given agents are {model_agent_i.unique_identifier,model_agent_j.unique_identifier} and the contributing agents are {self.barrier.contributing_agents}")
        
        self.model_agent_i = model_agent_i
        self.model_agent_j = model_agent_j
        
       # create a time variable
        entry_state_i = [casadi_tools.entry("state_"+str(self.model_agent_i.unique_identifier),shape=self.model_agent_i.state_vector.shape[0])]
        entry_state_j = [casadi_tools.entry("state_"+str(self.model_agent_j.unique_identifier),shape=self.model_agent_j.state_vector.shape[0])]
        entry_time    = [casadi_tools.entry('time',shape=1)]
        entry_input   = [casadi_tools.entry('input',shape=self.model_agent_i.input_vector.size1())]
    
        self.parameters = casadi_tools.struct_symMX(entry_state_i + entry_state_j + entry_time)
        self.x  = casadi_tools.struct_symMX(entry_state_i + entry_state_j + entry_time + entry_input)

    

        # compute Lie derivatives
        lie_fi_fun : ca.Function = self.model_agent_i.lie_derivative_f_function(self.barrier.function)
        lie_gi_fun : ca.Function = self.model_agent_i.lie_derivative_g_function(self.barrier.function) # contains now the control input as a variable
        barrier_time_derivative : ca.Function = self.barrier.partial_time_derivative
        
        if self.barrier.associated_alpha_function != None :
            alpha : ca.Function = self.barrier.associated_alpha_function
        else :
            raise Exception("The barrier function does not have an associated alpha function. PLease provide one")
        
      
        # compute reachable sets (these will be parameteric as a function of the initial state)
        Ai,bi = self.model_agent_i.reachable_set_A_and_b(self.parameters["state_"+str(self.model_agent_i.unique_identifier)])
        Aj,bj = self.model_agent_j.reachable_set_A_and_b(self.parameters["state_"+str(self.model_agent_j.unique_identifier)])

    
        # parameters of the optimization
        initial_parameters   = {"state_"+str(self.model_agent_i.unique_identifier):self.parameters["state_"+str(self.model_agent_i.unique_identifier)],
                                "state_"+str(self.model_agent_j.unique_identifier):self.parameters["state_"+str(self.model_agent_j.unique_identifier)],
                                "time"                        :self.parameters["time"]}
        
        # optimization variables of the optimization
        variables = {"state_"+str(self.model_agent_i.unique_identifier):self.x["state_"+str(self.model_agent_i.unique_identifier)],
                    "state_"+str(self.model_agent_j.unique_identifier) :self.x["state_"+str(self.model_agent_j.unique_identifier)],
                    "time"                         :self.x["time"]}
        
        # create the objective function  (note that n_ij = i_upsilon_ij + j_upsilon_ji and here you see only the i_upsilon_ij). The 0.5 it is because the Lipschitz constant is the sum of the two agents so they will compute half the contribution
        nlp_objective = -1* (           lie_fi_fun.call(initial_parameters)["value"] -         lie_fi_fun.call(variables)["value"] + 
                                (lie_gi_fun.call(initial_parameters)["value"] -         lie_gi_fun.call(variables)["value"])@self.x["input"]  + 
                            0.5*(barrier_time_derivative.call(initial_parameters)["value"] - barrier_time_derivative.call(variables)["value"]) +
                            0.5*(alpha(barrier.function.call(initial_parameters)["value"]) -  alpha(barrier.function.call(variables)["value"])) )

        
        # create the constraints
        state_i_constraint = Ai @ self.x["state_"+str(self.model_agent_i.unique_identifier)] - bi
        state_j_constraint = Aj @ self.x["state_"+str(self.model_agent_j.unique_identifier)] - bj
        input_constraint   = self.model_agent_i.input_constraints_A@self.x["input"] - self.model_agent_i.input_constraints_b
        time_constraints   = ca.vertcat(self.x["time"] - self.parameters["time"] - self.model_agent_i._time_step,  -(self.x["time"] - self.parameters["time"] ))  # delta < time_step -> delta - time_step<0   AND delta >0 -> -delta < 0
        constraints        = ca.vertcat(state_i_constraint,state_j_constraint,input_constraint,time_constraints )

       
        opts = {
        'print_iteration': False,    # Suppress iteration output
        'print_header': False,       # Suppress solver header
        'print_status': False,       # Suppress solver status
        'print_time': False,         # Suppress timing information
        'verbose_init': 0,           # Set verbosity level to minimum
        'expand':True,               # Expand the problem to make it faster
        'qpsol':'qpoases',
        'qpsol_options':{"printLevel":"none",'verbose': False,
                        "print_time":False,
                        "print_problem":False,}
        }
        
        
        with NoStdStreams():
            nlp = {'x':self.x , 'f':nlp_objective, 'g':constraints, 'p':self.parameters}
            S = ca.nlpsol('S', 'sqpmethod', nlp,opts)
            self.solver = ca.Function('solver',[ca.vertcat(self.x),ca.vertcat(self.parameters)],[S(x0=self.x,p=self.parameters,ubg=0)['f']])
        
    
    

    def generate_samples(self, current_time: float,
                               state_agent_i: np.ndarray,
                               state_agent_j: np.ndarray ,
                               num_samples: int = 10) -> list[Sample]:
        """
        Generates samples for upsilon optimization.

        Args:
            current_time (float): The current time.
            state_agent_i (np.ndarray): The state of agent i.
            state_agent_j (np.ndarray, optional): The state of agent j. Defaults to None.
            num_samples (int, optional): The number of samples to generate. Defaults to 10.

        Returns:
            samples (list[Sample]): The generated samples for the initial guess and the self.parameters. Note that all the samples witll have the same self.parameters. Only the intial guess it is different for each sample
        """
       
        samples = []
        
        # set paramseters
        p_value = self.parameters(0)
        p_value["state_" + str(self.model_agent_i.unique_identifier)] = state_agent_i
        p_value["state_" + str(self.model_agent_j.unique_identifier)] = state_agent_j
        p_value["time"] = current_time

        for sample in range(num_samples):
            
            # initialise initial guess
            x_guess = self.x(0)

            verticesi = self.model_agent_i.reachable_set_vertices(state_agent_i)
            verticesj = self.model_agent_j.reachable_set_vertices(state_agent_j)
            input_vertices = self.model_agent_i._input_constraints_vertices

            num_vertices_i = verticesi.shape[0]
            num_vertices_j = verticesj.shape[0]
            num_vertices_input = input_vertices.shape[0]
            
            # sampling from a polygone using the random convex combinaitons
            convex_coeffi = np.random.rand(num_vertices_i)
            convex_coeffi = convex_coeffi / np.sum(convex_coeffi )
            convex_coeffj = np.random.rand(num_vertices_j)
            convex_coeffj = convex_coeffj / np.sum(convex_coeffj)
            convex_coeff_input = np.random.rand(num_vertices_input)
            convex_coeff_input = convex_coeff_input / np.sum(convex_coeff_input )

            time_sample = current_time + np.random.rand(1) * self.model_agent_i._time_step

            # random initial state 1
            statei_sample = verticesi.T @ convex_coeffi
            statej_sample = verticesj.T @ convex_coeffj
            input_sample = input_vertices.T @ convex_coeff_input

            x_guess["state_" + str(self.model_agent_i.unique_identifier)] = statei_sample
            x_guess["state_" + str(self.model_agent_j.unique_identifier)] = statej_sample
            x_guess["time"] = time_sample
            x_guess["input"] = input_sample


            samples.append(Sample(x_guess, p_value))
        
        return samples
    



def compute_nu_collision_avoidance(max_velocity_i:float,max_velocity_j:float,position_i:np.ndarray,position_j:np.ndarray,alpha_factor:float,delta_t:float)-> float :
    """computes the maximum negative variation for a collsion avoidance barrier function among two single integrator systems.
    The solution is analytical
    
    b(x) = \|p1 -p2\|^2 - epsilon_r^2 -> collsion avoidance barrier function
    db_dx_1 = 2*(p1-p2) -> gradient of the barrier function
    db_dx_2 = -2*(p1-p2) -> gradient of the barrier function
    
    
    barrier constraint :
    db_dx_1 v_1 + db_dx_2 v_2 + alpha_factor * b(x) >= 0 
    2(p1-p2)(v_1-v_2) + alpha_factor * ( \|p1 -p2\|^2 - epsilon_r^2) >= 0
    
    -> We need to find the maximum negative variation of this (a  sort of one sided Lipschitz constant).
    -> We take another derivative (notes that v_1 and v_2 are constant)
    
    2 |v_1-v_2|^2 + alpha_factor * 2(p1-p2)(v_1-v_2) -> we need to find the minimum of this inside the reachable set of the two agents given a sampling interval (dt). Basically we find the one sided (negative) lipschitz constant of the pervious constraint
    
    p_1_tilde = p_1 + v_1 * dt 
    p_2_tilde = p_2 + v_2 * dt
    
    substituting back we get
    
    2(1+alpha_factor*dt)|dv|^2 + 2*alpha_factor (p1-p2)dv -> minimize this. it is non-convex but you can minimize this analytically. there is an analytical solution for the minimum

    Args:
        max_velocity_i (float): _description_
        max_velocity_j (float): _description_
        position_i (np.ndarray): _description_
        position_j (np.ndarray): _description_

    Returns:
        float: _description_
    """
    
    epsilon           = max_velocity_i + max_velocity_j #(maximum relative velocity (one going toward the other)
    relative_position = position_i.flatten() - position_j.flatten()
    dp                = np.sqrt(np.sum(relative_position**2))
    
    worse_dv = -alpha_factor/2 *  relative_position
    
    if np.sqrt(np.sum(worse_dv**2)) >= epsilon:
        lip = 2*epsilon*(epsilon-alpha_factor*dp)
    else :
        lip = -alpha_factor**2/2 * dp**2
       
    return lip*delta_t
  



class ImpactSolverLP:
    """
    This is a support class to basically solve a linear program. 
    min Lg^T u or max Lg^T u
    s.t A u <= b
    """
    def __init__(self,model:MathematicalDynamicModel) -> None:
        

        self.Lg    = ca.MX.sym("Lg",model.input_vector.size1()) # This is the parameteric Lie derivative. it will computed at every time outside this function and then it will be given as a parameter
        self.cost  = self.Lg.T @ model.input_vector # change of sign becaise you have to turn maximization into minimization
        
        # constraints
        A           = model.input_constraints_A
        b           = model.input_constraints_b
        constraints = A@model.input_vector - b # this is already an ca.MX that is a funciton of the control input
        
        with NoStdStreams():
            lp          = {'x':model.input_vector, 'f':self.cost, 'g':constraints,'p':self.Lg} # again we make it parameteric to avoid rebuilding the optimization program since the structure is always the same
            self.solver      = ca.qpsol('S', 'qpoases', lp,{"printLevel":"none"}) # create a solver object 
    
    def maximize(self,Lg:np.ndarray) -> np.ndarray:
        """This program will simply be solved as a function of the parameter. This avoids re bulding the optimization program every time"""
        
        return self.solver(p=-Lg,ubg=0)["x"]
    
    def minimize(self,Lg:np.ndarray) -> np.ndarray:
        """This program will simply be solved as a function of the parameter. This avoids re bulding the optimization program every time"""
        return self.solver(p=Lg,ubg=0)["x"]
    
    
    
