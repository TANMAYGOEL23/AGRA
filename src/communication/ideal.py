"""
Ideal and Fixed Lossy Communication Channels.
"""
import random
from collections import deque
from typing import List, Dict, Any
import numpy as np
from src.communication.channel import BaseCommChannel

class IdealChannel(BaseCommChannel):
    """Zero latency, 100% reliable broadcast conduit (Paper 1 assumption)."""
    def __init__(self):
        super().__init__(name="Ideal")

    def update_uav_communication(self, uav_list: List[Any], current_time: float, dt: float):
        for uav in uav_list:
            uav.comm_mode = "IDEAL"
            uav.packet_loss_prob = 0.0
            uav.effective_network_delay = 0.0
            uav.perceived_neighbors.clear()
            
            for other in uav_list:
                if other.uav_id == uav.uav_id:
                    continue
                self.total_packets_sent += 1
                self.total_packets_delivered += 1
                uav.perceived_neighbors[other.uav_id] = {
                    'pos': other.position.copy(),
                    'vel': other.velocity.copy(),
                    'timestamp': current_time
                }

class FixedLossyChannel(BaseCommChannel):
    """Fixed packet loss and latency channel."""
    def __init__(self, loss_rate: float = 0.2, delay: float = 0.05):
        super().__init__(name="FixedLossy")
        self.loss_rate = loss_rate
        self.delay = delay
        # In-flight packets queue: (delivery_time, src_id, dst_id, pos, vel)
        self.flight_buffer: List[Dict[str, Any]] = []

    def update_uav_communication(self, uav_list: List[Any], current_time: float, dt: float):
        # 1. Enqueue new broadcasts
        for src in uav_list:
            src.comm_mode = f"LOSS_{int(self.loss_rate*100)}%"
            src.packet_loss_prob = self.loss_rate
            src.effective_network_delay = self.delay
            
            for dst in uav_list:
                if dst.uav_id == src.uav_id:
                    continue
                self.total_packets_sent += 1
                
                # Check packet loss
                if random.random() < self.loss_rate:
                    self.total_packets_dropped += 1
                    continue
                    
                # Packet scheduled for delivery
                delivery_time = current_time + self.delay
                self.flight_buffer.append({
                    'delivery_time': delivery_time,
                    'src_id': src.uav_id,
                    'dst_id': dst.uav_id,
                    'pos': src.position.copy(),
                    'vel': src.velocity.copy()
                })

        # 2. Deliver ripe packets
        undelivered = []
        uav_map = {u.uav_id: u for u in uav_list}
        
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

    def reset(self):
        super().reset()
        self.flight_buffer.clear()
