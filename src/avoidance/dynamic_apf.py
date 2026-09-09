"""
Dynamic Artificial Potential Field (Dynamic APF) Controller.
Implements relative-velocity perpendicular repulsive force as formulated in Paper 1 Eq (5)-(7).
"""
import numpy as np
from src.avoidance.base import BaseAvoidanceController
from src.core.uav import UAV
from src.core.space import normalize_vector

class DynamicAPFController(BaseAvoidanceController):
    """
    Dynamic APF considers relative velocity vectors between UAV and dynamic obstacles/agents
    to create a normal repulsive force vector r_vn (Du et al. 2019 / Paper 1).
    """
    def __init__(self):
        super().__init__(name="Dynamic_APF")

    def compute_velocity(self, uav: UAV, dt: float, current_time: float) -> np.ndarray:
        # Eq (1): Attractive force
        a_vec = uav.goal - uav.position
        uav.attractive_force = a_vec.copy()
        
        rs = uav.config.sensing_radius
        r_p_total = np.zeros(3, dtype=float)
        r_vn_total = np.zeros(3, dtype=float)
        
        for neighbor_id, neighbor_data in uav.perceived_neighbors.items():
            neighbor_pos = neighbor_data['pos']
            neighbor_vel = neighbor_data.get('vel', np.zeros(3, dtype=float))
            
            r_vec = uav.position - neighbor_pos
            dist = np.linalg.norm(r_vec)
            
            if 0.0 < dist < rs:
                r_hat = normalize_vector(r_vec)
                r_p = (1.0 / max(0.25, dist ** 2)) * r_hat
                r_p_total += r_p
                
                # Eq (5): Relative velocity vector v_r = v_o - v_v
                v_r = neighbor_vel - uav.velocity
                v_r_norm = np.linalg.norm(v_r)
                
                if v_r_norm > 1e-4:
                    v_r_hat = v_r / v_r_norm
                    # Dot product between r_hat and v_r_hat
                    dot_prod = float(np.dot(r_hat, v_r_hat))
                    
                    # Eq (6): 0 < r_hat . v_r_hat < 1
                    if 0.0 < dot_prod < 1.0:
                        r_vn = (r_hat / dot_prod) - v_r_hat
                        r_vn_total += r_vn

        uav.repulsive_force = r_p_total.copy()
        uav.normal_repulsive_force = r_vn_total.copy()
        
        # Eq (7): Target velocity combining attractive, distance repulsive, and dynamic normal repulsive
        v_v = uav.config.k_pa * a_vec + uav.config.k_pp * r_p_total + uav.config.k_pv * r_vn_total
        return v_v
