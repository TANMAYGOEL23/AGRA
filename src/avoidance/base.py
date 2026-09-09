"""
Base Avoidance Controller Abstract Interface.
All avoidance algorithms (APF, Adaptive APF, Dynamic APF, ACACT) adhere to this contract.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple
from src.core.uav import UAV

class BaseAvoidanceController(ABC):
    """Abstract Base Class for Multi-UAV Collision Avoidance Controllers."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_velocity(self, uav: UAV, dt: float, current_time: float) -> np.ndarray:
        """
        Computes desired target velocity vector v_v for the given UAV
        based on goal target and perceived neighbors/obstacles.
        
        Returns:
            np.ndarray: Desired velocity vector (3D).
        """
        pass
        
    def reset(self):
        """Reset internal controller states if any."""
        pass
