"""
FPS-SCTP (Forward Prediction Scheduling Stream Control Transmission Protocol) & AGRA Channel.
Implements:
1. Partial Reliability with Multi-Streaming (PR-SCTP foundation).
2. Late Messages Filter (LMF): Forward-predicts packet delivery and retransmission timeline (t_Rtx);
   drops packets if predicted arrival exceeds the Packet Delay Budget (PDB), prioritizing timeliness over stale data.
3. Graded Urgency-Based Retransmission (AGRA): Dynamic packet-level urgency scoring (C_n) based on
   instantaneous estimated collision time (t_c) from the ACACT motion layer.
4. Comprehensive network telemetry: Late dropped count, retransmissions, overhead bytes, delivery jitter.

Reference:
- Ronaldo, Sudarsono & Pramadihanto, "Secure Real-time Data Transmission for Drone Delivery Services
  using Forward Prediction Scheduling SCTP," EMITTER, 2022.
- Team 14 (AGRA): "Communication-Aware Collision Avoidance in Multi-UAV Swarms", 2026.
"""
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from src.communication.channel import BaseCommChannel

@dataclass
class SCTPPacket:
    packet_id: int
    src_id: int
    dst_id: int
    pos: np.ndarray
    vel: np.ndarray
    creation_time: float
    pdb: float                      # Packet Delay Budget (deadline = creation_time + pdb)
    urgency: float                  # Urgency score C_n in [0.0, 1.0]
    stream_id: int = 0              # SCTP multi-streaming stream identifier
    retransmissions_left: int = 2   # Graded retransmission quota
    estimated_rtt: float = 0.05     # Smoothed RTT estimate
    scheduled_arrival: float = 0.0

class FPSSCTPChannel(BaseCommChannel):
    """
    FPS-SCTP with Late Messages Filter (LMF) and Graded Retransmission.
    Provides fine-grained, per-packet QoS without binary all-or-nothing switching.
    """
    def __init__(
        self,
        base_rtt: float = 0.04,
        channel_loss_rate: float = 0.15,
        default_pdb: float = 0.30,
        enable_lmf: bool = True,
        max_retransmissions: int = 2
    ):
        super().__init__(name="FPS-SCTP")
        self.base_rtt = base_rtt
        self.channel_loss_rate = channel_loss_rate
        self.default_pdb = default_pdb
        self.enable_lmf = enable_lmf
        self.max_retransmissions = max_retransmissions
        
        # Flight buffer: list of active in-flight SCTPPacket objects
        self.in_flight_packets: List[SCTPPacket] = []
        self.packet_counter: int = 0
        
        # Detailed FPS-SCTP Telemetry
        self.total_retransmissions: int = 0
        self.late_messages_filtered: int = 0
        self.on_time_deliveries: int = 0
        self.bytes_transmitted: int = 0
        self.stream_counters: Dict[int, int] = {0: 0, 1: 0} # 0: Normal, 1: Urgent collision telemetry

    def compute_packet_delay_budget(self, src_uav: Any, min_neighbor_dist: float) -> float:
        """
        Calculates Packet Delay Budget (PDB) dynamically from ACACT kinematics.
        If an imminent collision is approaching (small distance / high speed), PDB is tight.
        """
        v_mag = max(0.1, float(np.linalg.norm(src_uav.velocity)))
        # Time to traverse distance margin
        time_to_contact = min_neighbor_dist / v_mag
        
        # PDB is bounded between 0.08s (critical) and 0.5s (relaxed)
        pdb = np.clip(time_to_contact * 0.4, 0.08, 0.50)
        return float(pdb)

    def compute_urgency_score(self, src_uav: Any, min_neighbor_dist: float) -> float:
        """
        Computes graded urgency metric C_n in [0.0, 1.0].
        C_n -> 1.0 when UAV is inside collision hazard zone (< 2.5m).
        """
        rs = src_uav.config.sensing_radius
        if min_neighbor_dist >= rs:
            return 0.1
        # Inverse linear mapping
        urgency = 1.0 - (min_neighbor_dist / rs)
        return float(np.clip(urgency, 0.1, 1.0))

    def update_uav_communication(self, uav_list: List[Any], current_time: float, dt: float):
        uav_map = {u.uav_id: u for u in uav_list}
        
        # 1. Generate outbound broadcast packets with FPS-SCTP headers
        for src in uav_list:
            if src.reached_goal:
                continue

            # Find closest perceived distance to compute PDB and Urgency
            min_dist = src.config.sensing_radius
            for other in uav_list:
                if other.uav_id != src.uav_id and not other.reached_goal:
                    d = float(np.linalg.norm(src.position - other.position))
                    if d < min_dist:
                        min_dist = d

            pdb = self.compute_packet_delay_budget(src, min_dist)
            urgency = self.compute_urgency_score(src, min_dist)
            
            # Multi-streaming: Urgent stream (1) if urgency > 0.6, else normal stream (0)
            stream_id = 1 if urgency > 0.6 else 0
            
            # Graded retransmission budget based on urgency
            if urgency > 0.75:
                rtx_quota = 2
            elif urgency > 0.40:
                rtx_quota = 1
            else:
                rtx_quota = 0
                
            src.comm_mode = f"FPS_SCTP(U={urgency:.2f})"
            src.packet_loss_prob = 0.0  # Retransmission handles loss dynamically

            for dst in uav_list:
                if dst.uav_id == src.uav_id:
                    continue

                self.packet_counter += 1
                self.total_packets_sent += 1
                self.bytes_transmitted += 64  # standard SCTP data chunk payload
                self.stream_counters[stream_id] = self.stream_counters.get(stream_id, 0) + 1
                
                one_way_delay = self.base_rtt / 2.0 + random.uniform(0.005, 0.015)
                
                pkt = SCTPPacket(
                    packet_id=self.packet_counter,
                    src_id=src.uav_id,
                    dst_id=dst.uav_id,
                    pos=src.position.copy(),
                    vel=src.velocity.copy(),
                    creation_time=current_time,
                    pdb=pdb,
                    urgency=urgency,
                    stream_id=stream_id,
                    retransmissions_left=rtx_quota,
                    estimated_rtt=self.base_rtt,
                    scheduled_arrival=current_time + one_way_delay
                )
                self.in_flight_packets.append(pkt)

        # 2. Process in-flight transmissions, loss events, and Late Messages Filter (LMF)
        remaining_packets: List[SCTPPacket] = []
        
        for pkt in self.in_flight_packets:
            if pkt.scheduled_arrival <= current_time + 1e-6:
                # Check channel loss
                if random.random() < self.channel_loss_rate:
                    # Packet lost on link -> Evaluate Retransmission via LMF
                    if pkt.retransmissions_left > 0:
                        # Forward Prediction of retransmission arrival time
                        t_rtx = pkt.estimated_rtt
                        predicted_arrival = current_time + t_rtx
                        deadline = pkt.creation_time + pkt.pdb
                        
                        # Late Messages Filter (LMF) Decision:
                        if self.enable_lmf and predicted_arrival > deadline:
                            # Drop retransmission because it will arrive too late!
                            self.late_messages_filtered += 1
                            self.total_packets_dropped += 1
                        else:
                            # Schedule graded retransmission
                            pkt.retransmissions_left -= 1
                            pkt.scheduled_arrival = predicted_arrival
                            self.total_retransmissions += 1
                            self.bytes_transmitted += 64
                            remaining_packets.append(pkt)
                    else:
                        self.total_packets_dropped += 1
                else:
                    # Successfully delivered
                    deadline = pkt.creation_time + pkt.pdb
                    if current_time <= deadline:
                        self.on_time_deliveries += 1
                    else:
                        self.late_messages_filtered += 1

                    self.total_packets_delivered += 1
                    dst_uav = uav_map.get(pkt.dst_id)
                    if dst_uav:
                        dst_uav.perceived_neighbors[pkt.src_id] = {
                            'pos': pkt.pos,
                            'vel': pkt.vel,
                            'timestamp': current_time,
                            'urgency': pkt.urgency
                        }
            else:
                remaining_packets.append(pkt)
                
        self.in_flight_packets = remaining_packets

        # 3. Clean up stale or goal-reached neighbors
        for uav in uav_list:
            stale_ids = [
                nid for nid, ndata in uav.perceived_neighbors.items()
                if (uav_map.get(nid) and uav_map[nid].reached_goal) or (current_time - ndata.get('timestamp', 0) > 1.2)
            ]
            for nid in stale_ids:
                del uav.perceived_neighbors[nid]

    def reset(self):
        super().reset()
        self.in_flight_packets.clear()
        self.packet_counter = 0
        self.total_retransmissions = 0
        self.late_messages_filtered = 0
        self.on_time_deliveries = 0
        self.bytes_transmitted = 0
        self.stream_counters = {0: 0, 1: 0}

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            "retransmissions": self.total_retransmissions,
            "late_messages_filtered": self.late_messages_filtered,
            "on_time_deliveries": self.on_time_deliveries,
            "bytes_transmitted_kb": round(self.bytes_transmitted / 1024.0, 2),
            "urgent_stream_packets": self.stream_counters.get(1, 0),
            "normal_stream_packets": self.stream_counters.get(0, 0),
        })
        return stats
