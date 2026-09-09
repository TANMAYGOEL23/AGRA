"""
Delay- and Congestion-Aware Adaptive Communication Switching (DCACS).
Implements the protocol switching logic from Adithya & Kundu (Paper 2).
"""
import random
from typing import List, Dict, Any
import numpy as np
from src.communication.channel import BaseCommChannel

class DCACSChannel(BaseCommChannel):
    """
    DCACS Communication Channel.
    Decentrally updates each UAV's communication mode between FAST and RELIABLE
    based on individual link delay (delta=0.1s) and instantaneous neighbor density (gamma=3).
    """
    def __init__(self, delay_threshold: float = 0.1, congestion_threshold: int = 3, fast_loss_rate: float = 0.2):
        super().__init__(name="DCACS")
        self.delay_threshold = delay_threshold
        self.congestion_threshold = congestion_threshold
        self.fast_loss_rate = fast_loss_rate
        # In-flight packets queue: list of packet dicts
        self.flight_buffer: List[Dict[str, Any]] = []

    def update_uav_communication(self, uav_list: List[Any], current_time: float, dt: float):
        uav_map = {u.uav_id: u for u in uav_list}
        
        # 1. Update each UAV's instantaneous neighbor count and communication mode
        for uav in uav_list:
            # Count neighbors within sensing range r_s (or active perceived neighbors)
            rs = uav.config.sensing_radius
            neighbor_count = 0
            for other in uav_list:
                if other.uav_id != uav.uav_id:
                    dist = float(np.linalg.norm(uav.position - other.position))
                    if dist <= rs:
                        neighbor_count += 1

            # Paper 2 Eq (4): Mode selection rule
            # mi = RELIABLE if di > delta or ci >= gamma else FAST
            link_delay = uav.config.network_delay
            if link_delay > self.delay_threshold or neighbor_count >= self.congestion_threshold:
                uav.comm_mode = "RELIABLE"
                uav.packet_loss_prob = 0.0
                uav.effective_network_delay = link_delay
            else:
                uav.comm_mode = "FAST"
                uav.packet_loss_prob = self.fast_loss_rate
                uav.effective_network_delay = 0.0  # minimal baseline latency in FAST mode

        # 2. Transmit position/velocity packets according to active mode
        for src in uav_list:
            if src.reached_goal:
                continue
                
            for dst in uav_list:
                if dst.uav_id == src.uav_id:
                    continue
                    
                self.total_packets_sent += 1
                
                # Check for packet loss in FAST mode
                if src.comm_mode == "FAST" and random.random() < src.packet_loss_prob:
                    self.total_packets_dropped += 1
                    continue
                    
                # Calculate delivery latency
                latency = src.effective_network_delay if src.comm_mode == "RELIABLE" else 0.0
                delivery_time = current_time + latency
                
                self.flight_buffer.append({
                    'delivery_time': delivery_time,
                    'src_id': src.uav_id,
                    'dst_id': dst.uav_id,
                    'pos': src.position.copy(),
                    'vel': src.velocity.copy()
                })

        # 3. Deliver packets whose delivery_time <= current_time
        undelivered = []
        for packet in self.flight_buffer:
            if packet['delivery_time'] <= current_time + 1e-6:
                dst = uav_map.get(packet['dst_id'])
                if dst:
                    self.total_packets_delivered += 1
                    dst.perceived_neighbors[packet['src_id']] = {
                        'pos': packet['pos'],
                        'vel': packet['vel'],
                        'timestamp': current_time
                    }
            else:
                undelivered.append(packet)
                
        self.flight_buffer = undelivered

        # 4. Remove stale or goal-reached neighbors
        for uav in uav_list:
            stale_ids = [
                nid for nid, ndata in uav.perceived_neighbors.items()
                if (uav_map.get(nid) and uav_map[nid].reached_goal) or (current_time - ndata.get('timestamp', 0) > 1.5)
            ]
            for nid in stale_ids:
                del uav.perceived_neighbors[nid]

    def reset(self):
        super().reset()
        self.flight_buffer.clear()
