"""
Performance Evaluation Metrics: PTTR (Paper 1 & Paper 2), TTR, CTR, and Path Length.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from src.core.uav import UAV

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

def compute_trajectory_length(trajectory: List[np.ndarray]) -> float:
    """Computes total arc length traversed by the UAV."""
    if len(trajectory) < 2:
        return 0.0
    diffs = np.diff(np.array(trajectory), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def calculate_uav_metrics(uav: UAV, total_sim_time: float) -> UAVMetricsResult:
    """
    Evaluates a single UAV's trajectory and computes metrics from both papers.
    """
    t_travel = uav.completion_time if uav.completion_time is not None else total_sim_time
    t_travel = max(0.001, t_travel)
    t_col = uav.collision_time_accumulated
    
    d_initial = uav.initial_distance_to_goal()
    v_max = max(0.1, uav.config.v_max)
    traj_len = compute_trajectory_length(uav.trajectory)
    
    # Paper 1 Formulation (Min & Nam 2023, Eq 20):
    # PTTR = (d / v_max - t_col) / t_travel = TTR - CTR
    t_min = d_initial / v_max
    ttr = t_min / t_travel
    ctr = t_col / t_travel
    pttr_p1 = (t_min - t_col) / t_travel
    
    # Paper 2 Formulation (Adithya & Kundu, Eq 5):
    # PTTR_i = 1 / (1 + t_travel + t_col)
    pttr_p2 = 1.0 / (1.0 + t_travel + t_col)
    
    had_collision = t_col > 0.0
    
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
        had_collision=had_collision
    )

def calculate_swarm_summary(uav_list: List[UAV], total_sim_time: float) -> Dict[str, Any]:
    """Computes swarm-level aggregates matching the paper tables."""
    results = [calculate_uav_metrics(u, total_sim_time) for u in uav_list]
    
    mean_travel_time = np.mean([r.travel_time for r in results])
    mean_collision_time = np.mean([r.collision_time for r in results])
    max_collision_time = np.max([r.collision_time for r in results])
    mean_pttr_p1 = np.mean([r.pttr_paper1 for r in results])
    mean_pttr_p2 = np.mean([r.pttr_paper2 for r in results])
    mean_traj_len = np.mean([r.trajectory_length for r in results])
    
    return {
        "per_uav": results,
        "mean_travel_time": round(float(mean_travel_time), 2),
        "mean_collision_time": round(float(mean_collision_time), 2),
        "max_collision_time": round(float(max_collision_time), 2),
        "mean_trajectory_length": round(float(mean_traj_len), 2),
        "mean_pttr_paper1": round(float(mean_pttr_p1), 4),
        "mean_pttr_paper2": round(float(mean_pttr_p2), 5),
    }
