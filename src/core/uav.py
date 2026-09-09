"""
UAV Kinematic Model and Swarm Agent State Representation.
Supports both 2D (Paper 2) and 3D (Paper 1) kinematics and quadrotor dynamics.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Optional, Any
from src.core.space import normalize_vector, clamp_magnitude

@dataclass
class UAVConfig:
    uav_id: int
    name: str = ""
    v_max: float = 3.0           # Maximum velocity (m/s)
    sensing_radius: float = 7.0   # Sensor detection range r_s (m)
    collision_radius: float = 2.0 # Collision risk threshold (m)
    physical_radius: float = 0.5  # Physical drone footprint (m)
    k_pa: float = 1.0             # Attractive gain (kpa)
    k_pp: float = 1.5             # Repulsive gain (kpp)
    k_pv: float = 0.3             # Dynamic velocity gain (kpv)
    k_ip: float = 0.1             # Integral gain (kip)
    k_dp: float = 0.4             # Derivative gain (kdp)
    t_s: float = 2.0              # Target safety collision time threshold (s)
    r_ref: float = 1.5            # Oscillation cancellation distance threshold (m)
    network_delay: float = 0.05   # Assigned link delay (s)
    start_delay: float = 0.0      # Start delay offset (s) for stagger scenarios

class UAV:
    """
    Represents an autonomous UAV agent within the swarm.
    Maintains internal physics state, perceived neighbor knowledge,
    active communication mode, forces acting on it, and flight telemetry.
    """
    def __init__(self, config: UAVConfig, start_pos: np.ndarray, goal_pos: np.ndarray):
        self.config = config
        self.uav_id = config.uav_id
        self.name = config.name or f"UAV_{config.uav_id}"
        
        # Positions & Vectors in 3D (Z=0 for 2D scenarios)
        self.position = np.array(start_pos, dtype=float)
        self.initial_position = np.array(start_pos, dtype=float)
        self.goal = np.array(goal_pos, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.target_velocity = np.zeros(3, dtype=float)
        
        # Telemetry Forces for Visualizer & Monitor (as depicted in Paper 1 Fig 5)
        self.attractive_force = np.zeros(3, dtype=float)  # Green
        self.repulsive_force = np.zeros(3, dtype=float)   # Red
        self.normal_repulsive_force = np.zeros(3, dtype=float)
        self.contingency_active: bool = False
        
        # Communication State (DCACS Paper 2)
        self.comm_mode: str = "FAST"
        self.packet_loss_prob: float = 0.0
        self.effective_network_delay: float = config.network_delay
        
        # Flight History & Metrics Tracking
        self.trajectory: List[np.ndarray] = [self.position.copy()]
        self.velocity_history: List[np.ndarray] = [self.velocity.copy()]
        self.time_elapsed: float = 0.0
        self.collision_time_accumulated: float = 0.0
        self.reached_goal: bool = False
        self.completion_time: Optional[float] = None
        self.min_neighbor_distance: float = float("inf")
        self.collision_events_count: int = 0
        
        # Perceived state of other UAVs: {uav_id: {'pos': np.ndarray, 'vel': np.ndarray, 'timestamp': float}}
        self.perceived_neighbors: Dict[int, Dict[str, Any]] = {}
        
    def reset(self):
        """Resets UAV to its initial state."""
        self.position = self.initial_position.copy()
        self.velocity = np.zeros(3, dtype=float)
        self.target_velocity = np.zeros(3, dtype=float)
        self.attractive_force = np.zeros(3, dtype=float)
        self.repulsive_force = np.zeros(3, dtype=float)
        self.normal_repulsive_force = np.zeros(3, dtype=float)
        self.contingency_active = False
        self.comm_mode = "FAST"
        self.packet_loss_prob = 0.0
        self.effective_network_delay = self.config.network_delay
        self.trajectory = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_elapsed = 0.0
        self.collision_time_accumulated = 0.0
        self.reached_goal = False
        self.completion_time = None
        self.min_neighbor_distance = float("inf")
        self.collision_events_count = 0
        self.perceived_neighbors.clear()
        
    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))
        
    def initial_distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.initial_position))
        
    def update_motion(self, dt: float, target_velocity: np.ndarray, goal_tolerance: float = 0.25):
        """
        Integrates kinematics using a low-pass response / acceleration limiter
        representing quadrotor inertia and response time.
        """
        if self.time_elapsed < self.config.start_delay:
            self.time_elapsed += dt
            return

        if self.reached_goal:
            self.velocity = np.zeros(3, dtype=float)
            self.target_velocity = np.zeros(3, dtype=float)
            return

        # Check goal reached
        if self.distance_to_goal() <= goal_tolerance:
            self.reached_goal = True
            self.completion_time = self.time_elapsed
            self.velocity = np.zeros(3, dtype=float)
            self.target_velocity = np.zeros(3, dtype=float)
            self.trajectory.append(self.position.copy())
            return

        # Clamp target velocity by maximum allowable speed
        clamped_target_v = clamp_magnitude(target_velocity, self.config.v_max)
        self.target_velocity = clamped_target_v
        
        # Smooth velocity convergence towards target_velocity (PX4/PID velocity controller emulation)
        # Using a realistic time constant tau ~ 0.15s
        tau = 0.15
        alpha = min(1.0, dt / tau)
        self.velocity = (1.0 - alpha) * self.velocity + alpha * clamped_target_v
        
        # Position update
        self.position = self.position + self.velocity * dt
        self.time_elapsed += dt
        
        # Record trajectory
        self.trajectory.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())

    def record_collision_step(self, dt: float, min_dist_to_any_uav: float):
        """Accumulates collision area occupancy if within collision radius."""
        if min_dist_to_any_uav < self.min_neighbor_distance:
            self.min_neighbor_distance = min_dist_to_any_uav
            
        if min_dist_to_any_uav < self.config.collision_radius:
            self.collision_time_accumulated += dt
            self.collision_events_count += 1
