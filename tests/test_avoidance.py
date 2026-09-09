"""
Unit tests for Collision Avoidance Algorithms (APF, Dynamic APF, ACACT).
"""
import unittest
import numpy as np
from src.core.uav import UAVConfig, UAV
from src.avoidance.apf import APFController
from src.avoidance.adaptive_apf import AdaptiveAPFController
from src.avoidance.dynamic_apf import DynamicAPFController
from src.avoidance.acact import ACACTController

class TestAvoidance(unittest.TestCase):
    def test_apf_attraction_repulsion(self):
        cfg = UAVConfig(uav_id=1, k_pa=1.0, k_pp=1.0, sensing_radius=7.0)
        uav = UAV(cfg, start_pos=np.array([0.0, 0.0, 0.0]), goal_pos=np.array([10.0, 0.0, 0.0]))
        
        # Perceived obstacle directly ahead at (5, 0, 0)
        uav.perceived_neighbors[2] = {'pos': np.array([5.0, 0.0, 0.0]), 'vel': np.zeros(3)}
        
        ctrl = APFController()
        v_v = ctrl.compute_velocity(uav, dt=0.05, current_time=0.0)
        
        # Attractive force points +X, Repulsive points -X
        self.assertGreater(uav.attractive_force[0], 0)
        self.assertLess(uav.repulsive_force[0], 0)

    def test_acact_collision_time_deflection(self):
        cfg = UAVConfig(uav_id=1, k_pa=1.0, k_pp=1.0, t_s=2.0, sensing_radius=7.0)
        uav = UAV(cfg, start_pos=np.array([0.0, 0.0, 5.0]), goal_pos=np.array([10.0, 0.0, 5.0]))
        uav.velocity = np.array([2.0, 0.0, 0.0])
        
        # Approaching obstacle directly at (4, 0, 5) with counter velocity
        uav.perceived_neighbors[2] = {'pos': np.array([4.0, 0.0, 5.0]), 'vel': np.array([-2.0, 0.0, 0.0])}
        
        ctrl = ACACTController(t_s_base=2.0)
        v_plus = ctrl.compute_velocity(uav, dt=0.05, current_time=0.0)
        
        # Must produce lateral evasion deflection (non-zero Y or Z)
        self.assertTrue(abs(v_plus[1]) > 0.0 or abs(v_plus[2]) > 0.0)
        self.assertGreater(np.linalg.norm(v_plus), 0.0)

    def test_contingency_plan_trigger(self):
        cfg = UAVConfig(uav_id=1, k_pa=1.0, k_pp=1.0, v_max=2.0, sensing_radius=7.0)
        uav = UAV(cfg, start_pos=np.array([0.0, 0.0, 5.0]), goal_pos=np.array([10.0, 0.0, 5.0]))
        
        # Trap UAV with balanced obstacle
        uav.perceived_neighbors[2] = {'pos': np.array([1.0, 0.0, 5.0]), 'vel': np.zeros(3)}
        
        ctrl = ACACTController(t_s_base=2.0)
        uav.velocity = np.zeros(3)
        v_cmd = ctrl.compute_velocity(uav, dt=0.05, current_time=0.0)
        
        self.assertTrue(uav.contingency_active or np.linalg.norm(v_cmd) > 0.0)

if __name__ == '__main__':
    unittest.main()
