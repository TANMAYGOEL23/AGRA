"""
Adaptive Artificial Potential Field (Adaptive APF) Controller.
Adapts repulsive gain inversely with distance margin to soften aggressive deviations,
referenced in Paper 1 benchmark comparisons.
"""
import numpy as np
from src.avoidance.base import BaseAvoidanceController
from src.core.uav import UAV
from src.core.space import normalize_vector

class AdaptiveAPFController(BaseAvoidanceController):
    """
    Adaptive APF modulates repulsive potential based on distance ratio to target and obstacle,
    attempting to mitigate local minima and oscillation near goals.
    """
    def __init__(self):
        super().__init__(name="Adaptive_APF")

    def compute_velocity(self, uav: UAV, dt: float, current_time: float) -> np.ndarray:
        a_vec = uav.goal - uav.position
        uav.attractive_force = a_vec.copy()
        
        dist_to_goal = np.linalg.norm(a_vec)
        rs = uav.config.sensing_radius
        r_p_total = np.zeros(3, dtype=float)
        
        for neighbor_id, neighbor_data in uav.perceived_neighbors.items():
            neighbor_pos = neighbor_data['pos']
            r_vec = uav.position - neighbor_pos
            dist = np.linalg.norm(r_vec)
            
            if 0.0 < dist < rs:
                r_hat = normalize_vector(r_vec)
                adaptive_factor = (dist_to_goal ** 1.5) / (rs ** 1.5 + 1e-6)
                adaptive_factor = np.clip(adaptive_factor, 0.2, 1.5)
                r_p = adaptive_factor * (1.0 / max(0.25, dist ** 2)) * r_hat
                r_p_total += r_p
                
        uav.repulsive_force = r_p_total.copy()
        uav.normal_repulsive_force = np.zeros(3, dtype=float)
        
        v_v = uav.config.k_pa * a_vec + uav.config.k_pp * r_p_total
        return v_v
