import casadi as ca
import casadi as ca
import numpy as np
from abc import ABC, abstractmethod
from itertools import product
import os,sys

from   symaware.base.data import Identifier



def is_casadiMX(x):
    return isinstance(x, ca.MX)

def name_without_id(string):
    splitted_list = string.split('_')
    name = '_'.join(splitted_list[:-1])
    return name

def wrap_name(name:str,unique_identifier:Identifier):
    return name + "_" + str(unique_identifier)

def get_id_from_name(name:str)-> Identifier|None:
    splitted_name = name.split('_')
    if not ("state" in splitted_name):
        return None
    else :
        return int(splitted_name[1]) # index of the 
    

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
        
     

class PDdrone():
    """
    
    Linear LQR controller for hooverng drone to follow a given velocity profile
     
    State :
    theta   -> pitch angle of drone
    v_theta -> pitch angle velocity
    phi     -> roll angle of drone
    v_phi   -> roll angle velocity
    
    Input :
    tau_theta = pitch angle torque
    tau_phi   = roll angle torque
    """
    
    def __init__(self, I_x:float = 0.1, I_y:float = 0.1):
        
        # dynamics along the x and y axis
        # .
        # v_x = g*theta
        # .
        # theta = 1/I_y * tau_theta
        #
        # Hence
        #  ..
        #  v_x = g/I_y * tau_theta -> simple second order system with input tau_theta
        
        # SI units for everything
        self.g   = 9.81
        self.I_y = I_y
        self.I_x = I_x


        self._error_vx_prev = 0 # previous velocity error in vx
        self._error_vy_prev = 0 # previous velocity error in vy
        
        self._current_vx = 0
        self._current_vy = 0 
        
        # you can tune using matlab
        self._Kd = 0.195
        self._Kp = 0.000846
        
    def compute_input_torques(self,vx_ref:float,vy_ref:float) :
        """Simple PD controller to track a given velocity profile in the x-y component"""
        
        error_vx = vx_ref - self._current_vx
        error_vy = vy_ref - self._current_vy
        
        self.tau_theta = self._Kp * error_vx + self._Kd * (error_vx - self._error_vx_prev)
        self.tau_phi   = self._Kp * error_vy + self._Kd * (error_vy - self._error_vy_prev)
        
        self._error_vx_prev = error_vx
        self._error_vy_prev = error_vy
        
        torques = np.array([self.tau_phi,self.tau_theta, 0])
        force   = np.array([0,0,self.mass * self.g])
        
        # These are torques
        return force, torques
    
        