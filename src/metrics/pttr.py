"""
Performance Evaluation Metrics: PTTR (Paper 1 & Paper 2), TTR, CTR, and Path Length.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from src.core.uav import UAV
from src.metrics.energy import UAVEnergyModel

@dataclass
class UAVMetricsResult:
    uav_id: int
    name: str
    travel_time: float
    collision_time: float
    trajectory_length: float
    shortest_distance: float
    min_separation: float
    v_max: float
    assigned_delay: float
    final_comm_mode: str
    
    # Paper 1 Metric
    pttr_paper1: float
    ttr: float
    ctr: float
    
    # Paper 2 Metric
    pttr_paper2: float
    
    # Collision Flag
    had_collision: bool
    
    # Energy & Communication Metrics
    energy_joules: float = 0.0
    battery_mah: float = 0.0
    avg_power_watts: float = 0.0
    packets_sent: int = 0
    packets_lost: int = 0
    packet_loss_rate_pct: float = 0.0
    retransmissions: int = 0
    bandwidth_used_kb: float = 0.0
    bandwidth_rate_kbps: float = 0.0

def compute_trajectory_length(trajectory: List[np.ndarray]) -> float:
    """Computes total arc length traversed by the UAV."""
    if len(trajectory) < 2:
        return 0.0
    diffs = np.diff(np.array(trajectory), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def calculate_uav_metrics(
    uav: UAV,
    total_sim_time: float,
    comm_stats: Optional[Dict[str, Any]] = None,
    total_uavs_count: int = 1
) -> UAVMetricsResult:
    """
    Evaluates a single UAV's trajectory and computes metrics across motion, communication, and energy.
    """
    t_travel = uav.completion_time if uav.completion_time is not None else total_sim_time
    t_travel = max(0.001, t_travel)
    t_col = uav.collision_time_accumulated
    
    d_initial = uav.initial_distance_to_goal()
    v_max = max(0.1, uav.config.v_max)
    traj_len = compute_trajectory_length(uav.trajectory)
    
    # Paper 1 Formulation (Min & Nam 2023, Eq 20):
    t_min = d_initial / v_max
    ttr = t_min / t_travel
    ctr = t_col / t_travel
    pttr_p1 = (t_min - t_col) / t_travel
    
    # Paper 2 Formulation (Adithya & Kundu, Eq 5):
    pttr_p2 = 1.0 / (1.0 + t_travel + t_col)
    had_collision = t_col > 0.0

    # Energy calculations
    energy_model = UAVEnergyModel()
    n_other = max(1, total_uavs_count - 1)
    
    # Estimate per-uav packet counts
    steps = len(uav.trajectory)
    pkt_sent = int(steps * n_other)
    
    rtx = 0
    pkt_lost = 0
    if comm_stats:
        total_s = max(1, comm_stats.get("packets_sent", 1))
        rtx_total = comm_stats.get("retransmissions", 0)
        drop_total = comm_stats.get("packets_dropped", 0)
        # Proportion for this UAV
        rtx = int(rtx_total / max(1, total_uavs_count))
        pkt_lost = int(drop_total / max(1, total_uavs_count))
    else:
        pkt_lost = int(pkt_sent * uav.packet_loss_prob)

    energy_data = energy_model.calculate_mission_energy(
        trajectory=uav.trajectory,
        velocity_history=uav.velocity_history,
        total_time=t_travel,
        dt=0.05,
        packets_sent=pkt_sent,
        packets_recv=pkt_sent,
        retransmissions=rtx
    )

    plr_pct = (pkt_lost / max(1, pkt_sent + rtx)) * 100.0
    bytes_used = (pkt_sent + rtx) * 64
    bw_kb = bytes_used / 1024.0
    bw_rate_kbps = (bytes_used * 8.0 / 1000.0) / max(0.01, t_travel)

    return UAVMetricsResult(
        uav_id=uav.uav_id,
        name=uav.name,
        travel_time=round(t_travel, 2),
        collision_time=round(t_col, 2),
        trajectory_length=round(traj_len, 2),
        shortest_distance=round(d_initial, 2),
        min_separation=round(uav.min_neighbor_distance if uav.min_neighbor_distance != float("inf") else 999.0, 2),
        v_max=v_max,
        assigned_delay=uav.config.network_delay,
        final_comm_mode=uav.comm_mode,
        pttr_paper1=round(pttr_p1, 4),
        ttr=round(ttr, 4),
        ctr=round(ctr, 4),
        pttr_paper2=round(pttr_p2, 5),
        had_collision=had_collision,
        energy_joules=energy_data["total_energy_joules"],
        battery_mah=energy_data["battery_consumed_mah"],
        avg_power_watts=energy_data["avg_power_watts"],
        packets_sent=pkt_sent,
        packets_lost=pkt_lost,
        packet_loss_rate_pct=round(plr_pct, 2),
        retransmissions=rtx,
        bandwidth_used_kb=round(bw_kb, 2),
        bandwidth_rate_kbps=round(bw_rate_kbps, 2)
    )

def calculate_swarm_summary(
    uav_list: List[UAV],
    total_sim_time: float,
    comm_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Computes swarm-level aggregates for all metrics."""
    n_uavs = len(uav_list)
    results = [calculate_uav_metrics(u, total_sim_time, comm_stats, n_uavs) for u in uav_list]
    
    mean_travel_time = np.mean([r.travel_time for r in results])
    mean_collision_time = np.mean([r.collision_time for r in results])
    max_collision_time = np.max([r.collision_time for r in results])
    mean_pttr_p1 = np.mean([r.pttr_paper1 for r in results])
    mean_pttr_p2 = np.mean([r.pttr_paper2 for r in results])
    mean_traj_len = np.mean([r.trajectory_length for r in results])
    
    total_energy_j = np.sum([r.energy_joules for r in results])
    total_battery_mah = np.sum([r.battery_mah for r in results])
    mean_power_w = np.mean([r.avg_power_watts for r in results])
    
    total_pkts_sent = np.sum([r.packets_sent for r in results])
    total_pkts_lost = np.sum([r.packets_lost for r in results])
    total_retransmissions = np.sum([r.retransmissions for r in results])
    total_bw_kb = np.sum([r.bandwidth_used_kb for r in results])
    mean_plr_pct = (total_pkts_lost / max(1, total_pkts_sent + total_retransmissions)) * 100.0
    
    return {
        "per_uav": results,
        "swarm_size": n_uavs,
        "mean_travel_time": round(float(mean_travel_time), 2),
        "mean_collision_time": round(float(mean_collision_time), 2),
        "max_collision_time": round(float(max_collision_time), 2),
        "mean_trajectory_length": round(float(mean_traj_len), 2),
        "mean_pttr_paper1": round(float(mean_pttr_p1), 4),
        "mean_pttr_paper2": round(float(mean_pttr_p2), 5),
        "total_energy_joules": round(float(total_energy_j), 2),
        "total_battery_mah": round(float(total_battery_mah), 2),
        "mean_power_watts": round(float(mean_power_w), 2),
        "total_packets_sent": int(total_pkts_sent),
        "total_packets_lost": int(total_pkts_lost),
        "total_retransmissions": int(total_retransmissions),
        "packet_loss_rate_pct": round(float(mean_plr_pct), 2),
        "total_bandwidth_kb": round(float(total_bw_kb), 2),
        "comm_overhead_pct": round(float((total_retransmissions / max(1, total_pkts_sent)) * 100.0), 2)
    }
