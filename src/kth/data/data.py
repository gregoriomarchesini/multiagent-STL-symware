import numpy as np
from   enum import Enum
from   dataclasses import dataclass
from typing import TYPE_CHECKING
from kth.stl.graphs import LeadershipToken

from symaware.base import (
    KnowledgeDatabase,
    Message,
)

if TYPE_CHECKING:
    from stl.stl import StlTask
    from kth.stl.graphs import LeadershipToken



@dataclass(frozen=True)
class StateMessage(Message):
    """A message that contains the state of an agent"""
    state: np.ndarray
    time_stamp: float = 0.0


# dictionary
class STLKnowledgeDatabase(KnowledgeDatabase):
    stl_tasks          : list["StlTask"]              # stores the task for an agent in a list
    leadership_tokens  : dict[int,"LeadershipToken"]    # stores the leadership tokens for the agent
    initial_time       : float                        # stores the initial time of the agent
    
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
    type      : ControlMessageType
    time_stamp: float = 0.0
    value     : float = 0.0
    


