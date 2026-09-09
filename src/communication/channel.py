"""
Base Abstract Communication Channel.
Defines the protocol and channel abstraction for swarm inter-UAV messaging.
Allows arbitrary network layers (Ideal, DCACS, 802.11p, 5G Sidelink, CSMA, TDMA) to be plugged in.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class BaseCommChannel(ABC):
    """
    Abstract communication channel representing the wireless medium between swarm UAVs.
    """
    def __init__(self, name: str):
        self.name = name
        self.total_packets_sent: int = 0
        self.total_packets_dropped: int = 0
        self.total_packets_delivered: int = 0
        
    @abstractmethod
    def update_uav_communication(self, uav_list: List[Any], current_time: float, dt: float):
        """
        Processes message broadcasts and updates perceived neighbor lists
        for all UAVs in the swarm according to the channel physics & protocol rules.
        """
        pass
        
    def reset(self):
        self.total_packets_sent = 0
        self.total_packets_dropped = 0
        self.total_packets_delivered = 0
        
    def get_stats(self) -> Dict[str, Any]:
        drop_rate = (self.total_packets_dropped / max(1, self.total_packets_sent)) * 100.0
        return {
            "channel_type": self.name,
            "packets_sent": self.total_packets_sent,
            "packets_delivered": self.total_packets_delivered,
            "packets_dropped": self.total_packets_dropped,
            "drop_rate_pct": drop_rate
        }
