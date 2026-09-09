"""
ACACT: Adaptive Collision Avoidance Algorithm Based on Estimated Collision Time.
Implements the full algorithm described in Min & Nam (IEEE Access 2023) and Adithya & Kundu.
Includes:
- Projected Target Velocity & Estimated Collision Time calculation (Eq 8-15)
- Oscillation Cancellation near Target (Eq 16)
- Contingency Plan Local Minima Escape (Algorithm 1)
- Swarm Multi-Agent Collision Avoidance (Algorithm 2, Eq 18-19)
- Delay-adaptive safety threshold integration (Paper 2 Eq 2)
"""
import numpy as np
import random
from src.avoidance.base import BaseAvoidanceController
from src.core.uav import UAV
from src.core.space import normalize_vector

class ACACTController(BaseAvoidanceController):
    """
    ACACT Avoidance Controller with adaptive target velocity adjustment.
    Ensures UAV maintains operational cruise speed while smoothly deflecting
    trajectories according to predicted collision timelines.
    """
    def __init__(self, t_s_base: float = 2.0, r_ref: float = 1.5):
        super().__init__(name="ACACT")
        self.t_s_base = t_s_base
        self.r_ref = r_ref
        self.contingency_target = None
        self.contingency_timer = 0.0

    def compute_velocity(self, uav: UAV, dt: float, current_time: float) -> np.ndarray:
        # Eq (1): Attractive force vector pulling to goal
        a_vec = uav.goal - uav.position
        a_norm = float(np.linalg.norm(a_vec))
        uav.attractive_force = a_vec.copy()
        
        rs = uav.config.sensing_radius
        r_p_total = np.zeros(3, dtype=float)
        r_vn_total = np.zeros(3, dtype=float)
        
        # Track closest neighbor / imminent threat (Eq 19)
        min_dist = float("inf")
        r_min_vec = None
        r_min_vel = None
        
        for neighbor_id, neighbor_data in uav.perceived_neighbors.items():
            neighbor_pos = neighbor_data['pos']
            neighbor_vel = neighbor_data.get('vel', np.zeros(3, dtype=float))
            
            r_vec = uav.position - neighbor_pos
            dist = float(np.linalg.norm(r_vec))
            
            if 0.0 < dist < rs:
                r_hat = normalize_vector(r_vec)
                # Distance repulsive force: 1 / dist^2 (Paper 2 Eq 1, clamped for stability)
                r_p = (1.0 / max(0.25, dist ** 2)) * r_hat
                r_p_total += r_p
                
                # Eq (5): Relative velocity vector v_r = v_o - v_v
                v_r = neighbor_vel - uav.velocity
                v_r_norm = float(np.linalg.norm(v_r))
                
                r_vn = np.zeros(3, dtype=float)
                if v_r_norm > 1e-4:
                    v_r_hat = v_r / v_r_norm
                    dot_prod = float(np.dot(r_hat, v_r_hat))
                    # Eq (6)
                    if 0.0 < dot_prod < 1.0:
                        r_vn = (r_hat / dot_prod) - v_r_hat
                        r_vn_total += r_vn

                # Eq (19): Identify r_min
                if dist < min_dist:
                    min_dist = dist
                    r_min_vec = r_vec
                    r_min_vel = neighbor_vel

        uav.repulsive_force = r_p_total.copy()
        uav.normal_repulsive_force = r_vn_total.copy()
        
        r_p_norm = float(np.linalg.norm(r_p_total))
        r_vn_norm = float(np.linalg.norm(r_vn_total))
        
        # Eq (16): Oscillation Cancellation near Target
        if a_norm < self.r_ref and r_p_norm > 0.0 and r_vn_norm < 0.2:
            v_v = uav.config.k_pa * a_vec
        else:
            v_v = (uav.config.k_pa * a_vec + 
                   uav.config.k_pp * r_p_total + 
                   uav.config.k_pv * r_vn_total)

        v_v_norm = float(np.linalg.norm(v_v))

        # Check / Handle Contingency Plan (Algorithm 1: Local Minima Escape)
        if self.contingency_timer > 0:
            self.contingency_timer -= dt
            if self.contingency_target is not None:
                escape_vec = self.contingency_target - uav.position
                if np.linalg.norm(escape_vec) > 0.2:
                    return normalize_vector(escape_vec) * min(uav.config.v_max, 1.5)
                else:
                    self.contingency_timer = 0.0
                    self.contingency_target = None
                    uav.contingency_active = False

        if a_norm > 1.0 and v_v_norm < 0.5 and min_dist < rs:
            # Trigger Algorithm 1 Contingency Plan
            uav.contingency_active = True
            escape_pos = self._compute_contingency_escape(uav, rs)
            self.contingency_target = escape_pos
            self.contingency_timer = 1.2  # Execute escape for 1.2 seconds
            escape_vec = escape_pos - uav.position
            return normalize_vector(escape_vec) * min(uav.config.v_max, 1.5)

        # If no imminent obstacle, return standard target velocity
        if r_min_vec is None or min_dist >= rs or v_v_norm < 1e-4:
            return v_v

        # Delay-adaptive safety threshold (Paper 2 Eq 2: T_s = T_s,base + d_i)
        ts_eff = uav.config.t_s + uav.effective_network_delay
        
        # Eq (10): Projected target velocity onto obstacle direction
        # r_min_vec points from obstacle to UAV; closing in occurs when dot(v_v, r_min_vec) < 0
        r_hat = normalize_vector(r_min_vec)
        dot_vr = float(np.dot(v_v, r_min_vec))
        
        if dot_vr < 0:
            # Component of v_v pointing towards obstacle
            v_v_p = (dot_vr / (min_dist ** 2)) * r_min_vec
            v_p_mag = float(np.linalg.norm(v_v_p))
            
            # Eq (12): Estimated collision time t_c = ||r|| / ||v_p||
            tc = (min_dist ** 2) / (abs(dot_vr) + 1e-6)
            
            # If predicted collision time is within safety threshold:
            if tc < ts_eff:
                # Eq (13): Reduce approach velocity to ||r|| / t_s
                v_p_plus_mag = min(min_dist / max(0.01, ts_eff), v_p_mag * 0.9)
                v_p_hat = normalize_vector(v_v_p)
                
                # Eq (14): Orthogonal / Vertical deflection velocity vector
                v_v_vert = v_v - v_v_p
                vert_norm = float(np.linalg.norm(v_v_vert))
                
                if vert_norm < 1e-4:
                    # Trajectory is collinear; generate robust orthogonal evasion vector
                    if abs(r_hat[0]) > 1e-3 or abs(r_hat[1]) > 1e-3:
                        v_v_vert = np.array([-r_hat[1], r_hat[0], 0.0])
                    else:
                        v_v_vert = np.array([1.0, 0.0, 0.0])
                    v_v_vert_hat = normalize_vector(v_v_vert)
                else:
                    v_v_vert_hat = v_v_vert / vert_norm
                    
                # Eq (15): Updated target velocity preserving total magnitude
                rem_mag_sq = max(0.0, (v_v_norm ** 2) - (v_p_plus_mag ** 2))
                v_v_plus = (v_p_plus_mag * v_p_hat) + (np.sqrt(rem_mag_sq) * v_v_vert_hat)
                return v_v_plus

        return v_v

    def _compute_contingency_escape(self, uav: UAV, rs: float) -> np.ndarray:
        """Implements Algorithm 1: Contingency escape behind current direction."""
        # Backward angle away from obstacles
        rphi = random.uniform(0, np.pi / 2)
        rtheta = random.uniform(0, 2 * np.pi)
        
        # Escape offset in backward direction (80% of sensor radius)
        dx = -0.8 * rs * np.cos(rphi)
        dy = 0.8 * rs * np.sin(rtheta) * np.cos(rtheta)
        dz = 0.8 * rs * np.sin(rtheta) * np.sin(rtheta)
        
        # Keep within reasonable height bounds
        escape_point = uav.position + np.array([dx, dy, dz])
        escape_point[2] = max(0.5, escape_point[2])
        return escape_point
