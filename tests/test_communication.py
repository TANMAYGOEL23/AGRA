"""
Unit tests for Communication Channels and DCACS Adaptive Mode Switching.
"""
import unittest
import numpy as np
from src.core.uav import UAVConfig, UAV
from src.communication.ideal import IdealChannel, FixedLossyChannel
from src.communication.dcacs import DCACSChannel

class TestCommunication(unittest.TestCase):
    def test_ideal_channel_delivery(self):
        cfg1 = UAVConfig(uav_id=1, network_delay=0.0)
        cfg2 = UAVConfig(uav_id=2, network_delay=0.0)
        u1 = UAV(cfg1, np.array([0.0, 0.0, 0.0]), np.array([5.0, 5.0, 0.0]))
        u2 = UAV(cfg2, np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        
        channel = IdealChannel()
        channel.update_uav_communication([u1, u2], current_time=0.0, dt=0.05)
        
        self.assertIn(2, u1.perceived_neighbors)
        self.assertIn(1, u2.perceived_neighbors)
        self.assertEqual(u1.comm_mode, "IDEAL")

    def test_dcacs_novelty_a_delay_adaptive(self):
        # Delay-based switching: delay > 0.1s selects RELIABLE, delay <= 0.1s selects FAST in uncongested environment
        cfg_fast = UAVConfig(uav_id=1, network_delay=0.05)
        cfg_reliable = UAVConfig(uav_id=2, network_delay=0.20)
        
        u1 = UAV(cfg_fast, np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 0.0]))
        u2 = UAV(cfg_reliable, np.array([10.0, 0.0, 0.0]), np.array([0.0, 10.0, 0.0]))
        
        channel = DCACSChannel(delay_threshold=0.1, congestion_threshold=3)
        channel.update_uav_communication([u1, u2], current_time=0.0, dt=0.05)
        
        self.assertEqual(u1.comm_mode, "FAST")
        self.assertEqual(u1.packet_loss_prob, 0.2)
        self.assertEqual(u2.comm_mode, "RELIABLE")
        self.assertEqual(u2.packet_loss_prob, 0.0)

    def test_dcacs_novelty_b_congestion_adaptive(self):
        # Congestion-based switching: 5 UAVs clustered closely should force all to RELIABLE even with delay=0.05s
        uavs = []
        for i in range(5):
            cfg = UAVConfig(uav_id=i+1, network_delay=0.05)
            uavs.append(UAV(cfg, np.array([1.0 + i*0.2, 1.0, 0.0]), np.array([10.0, 10.0, 0.0])))
            
        channel = DCACSChannel(delay_threshold=0.1, congestion_threshold=3)
        channel.update_uav_communication(uavs, current_time=0.0, dt=0.05)
        
        # All 5 UAVs see >= 3 neighbors and must switch to RELIABLE
        for u in uavs:
            self.assertEqual(u.comm_mode, "RELIABLE")
            self.assertEqual(u.packet_loss_prob, 0.0)

if __name__ == '__main__':
    unittest.main()
