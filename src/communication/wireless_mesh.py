"""
Extensible Wireless Mesh, 802.11p / 5G NR Sidelink & MAC Channel.
Provides an extensible simulation foundation for physical layer path-loss,
SINR-based packet error rate (PER), CSMA/CA contention, and TDMA scheduling.
"""
import random
from typing import List, Dict, Any, Optional
import numpy as np
from src.communication.channel import BaseCommChannel

class WirelessMeshChannel(BaseCommChannel):
    """
    Simulates radio frequency (RF) wireless communication with path loss,
    fading, and MAC contention (CSMA/CA or TDMA).
    """
    def __init__(
        self,
        tx_power_dbm: float = 20.0,
        carrier_freq_ghz: float = 5.9,     # 5.9 GHz for DSRC / 802.11p & C-V2X
        path_loss_exponent: float = 2.5,
        noise_floor_dbm: float = -95.0,
        mac_protocol: str = "CSMA_CA",     # 'CSMA_CA', 'TDMA', or 'SLOTTED_ALOHA'
        channel_bandwidth_mhz: float = 10.0
    ):
        super().__init__(name=f"WirelessMesh_{mac_protocol}")
        self.tx_power_dbm = tx_power_dbm
        self.carrier_freq_ghz = carrier_freq_ghz
        self.path_loss_exponent = path_loss_exponent
        self.noise_floor_dbm = noise_floor_dbm
        self.mac_protocol = mac_protocol
        self.channel_bandwidth_mhz = channel_bandwidth_mhz
        self.flight_buffer: List[Dict[str, Any]] = []

    def compute_received_power(self, distance: float) -> float:
        """Log-distance path loss model in dBm."""
        d = max(1.0, distance)
        # Free-space path loss at reference 1m at 5.9GHz is approx 47.8 dB
        pl_0 = 20 * np.log10(self.carrier_freq_ghz * 1e9) + 20 * np.log10(4 * np.pi / 3e8) - 147.55
        pl_d = pl_0 + 10 * self.path_loss_exponent * np.log10(d)
        rx_power = self.tx_power_dbm - pl_d
        return rx_power

    def compute_packet_error_rate(self, distance: float, interference_power_dbm: float = -120.0) -> float:
        """Computes BER/PER from Signal-to-Interference-plus-Noise Ratio (SINR)."""
        rx_power_dbm = self.compute_received_power(distance)
        sinr_db = rx_power_dbm - max(self.noise_floor_dbm, interference_power_dbm)
        
        # Sigmoid PER curve approximation for QPSK / 16-QAM
        if sinr_db > 15.0:
            return 0.01
        elif sinr_db < 0.0:
            return 0.95
        else:
            return float(1.0 / (1.0 + np.exp((sinr_db - 7.0) / 1.5)))

    def update_uav_communication(self, uav_list: List[Any], current_time: float, dt: float):
        uav_map = {u.uav_id: u for u in uav_list}
        
        for src in uav_list:
            if src.reached_goal:
                continue
                
            src.comm_mode = f"MESH_{self.mac_protocol}"
            
            for dst in uav_list:
                if dst.uav_id == src.uav_id:
                    continue
                    
                dist = float(np.linalg.norm(src.position - dst.position))
                per = self.compute_packet_error_rate(dist)
                
                self.total_packets_sent += 1
                
                # MAC Contention collision model
                if self.mac_protocol == "CSMA_CA":
                    # Higher density increases backoff / collision probability
                    density_penalty = min(0.4, len(src.perceived_neighbors) * 0.05)
                    per = min(0.99, per + density_penalty)
                    
                if random.random() < per:
                    self.total_packets_dropped += 1
                    continue
                    
                # Transmission delay: propagation (negligible) + transmission + queueing
                tx_delay = 0.005 + (0.02 if self.mac_protocol == "CSMA_CA" else 0.01)
                delivery_time = current_time + tx_delay
                
                self.flight_buffer.append({
                    'delivery_time': delivery_time,
                    'src_id': src.uav_id,
                    'dst_id': dst.uav_id,
                    'pos': src.position.copy(),
                    'vel': src.velocity.copy()
                })

        # Deliver packets
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

    def reset(self):
        super().reset()
        self.flight_buffer.clear()
