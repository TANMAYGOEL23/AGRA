"""
Traditional Artificial Potential Field (APF) Controller (Khatib 1986).
Paper 1 Equations (1)-(4), (18).
"""
import numpy as np
from src.avoidance.base import BaseAvoidanceController
from src.core.uav import UAV
from src.core.space import normalize_vector

class APFController(BaseAvoidanceController):
    """
    Standard APF algorithm combining attractive force towards goal
    and inverse-square repulsive force away from obstacles/neighbors within r_s.
    """
    def __init__(self):
        super().__init__(name="APF")

    def compute_velocity(self, uav: UAV, dt: float, current_time: float) -> np.ndarray:
        # Eq (1): Attractive force vector
        a_vec = uav.goal - uav.position
        uav.attractive_force = a_vec.copy()
        
        # Eq (2), (3), (18): Swarm Repulsive forces sum
        r_p_total = np.zeros(3, dtype=float)
        rs = uav.config.sensing_radius
        
        for neighbor_id, neighbor_data in uav.perceived_neighbors.items():
            neighbor_pos = neighbor_data['pos']
            r_vec = uav.position - neighbor_pos  # Direction moving away from neighbor
            dist = np.linalg.norm(r_vec)
            
            if 0.0 < dist < rs:
                r_hat = normalize_vector(r_vec)
                r_p = (1.0 / max(0.25, dist ** 2)) * r_hat
                r_p_total += r_p
                
        uav.repulsive_force = r_p_total.copy()
        uav.normal_repulsive_force = np.zeros(3, dtype=float)
        
        # Eq (4): Target velocity
        v_v = uav.config.k_pa * a_vec + uav.config.k_pp * r_p_total
        return v_v
