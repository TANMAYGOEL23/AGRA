"""
Discrete-Event / Fixed-Step Multi-UAV Simulation Engine.
Orchestrates UAV Kinematics, Collision Avoidance, Wireless Network Stack, and Telemetry.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from src.core.uav import UAV
from src.avoidance.base import BaseAvoidanceController
from src.communication.channel import BaseCommChannel
from src.scenarios.scenario_loader import Scenario
from src.metrics.pttr import calculate_swarm_summary

class SimulationEngine:
    """
    Core engine managing the simulation lifecycle.
    Decoupled architecture: Any Avoidance Controller + Any Communication Channel + Any Scenario.
    """
    def __init__(
        self,
        scenario: Scenario,
        controller: BaseAvoidanceController,
        channel: BaseCommChannel,
        goal_tolerance: float = 0.3
    ):
        self.scenario = scenario
        self.controller = controller
        self.channel = channel
        self.goal_tolerance = goal_tolerance
        
        self.uavs: List[UAV] = scenario.uavs
        self.current_time: float = 0.0
        self.step_count: int = 0
        self.dt: float = scenario.dt
        self.is_finished: bool = False
        self.history_snapshots: List[Dict[str, Any]] = []

    def reset(self):
        self.current_time = 0.0
        self.step_count = 0
        self.is_finished = False
        self.history_snapshots.clear()
        
        for uav in self.uavs:
            uav.reset()
            
        self.controller.reset()
        self.channel.reset()

    def step(self) -> bool:
        """
        Executes a single simulation tick:
        1. Wireless network communication update (loss, delay, DCACS mode switching)
        2. Collision avoidance velocity generation for each active UAV
        3. Kinematic integration and collision detection
        4. Check completion criteria
        
        Returns:
            bool: True if simulation is still ongoing, False if finished.
        """
        if self.is_finished:
            return False

        # 1. Update communication layer
        self.channel.update_uav_communication(self.uavs, self.current_time, self.dt)

        # 2. Compute velocity command for each UAV
        target_velocities: Dict[int, np.ndarray] = {}
        for uav in self.uavs:
            if not uav.reached_goal and self.current_time >= uav.config.start_delay:
                v_target = self.controller.compute_velocity(uav, self.dt, self.current_time)
                target_velocities[uav.uav_id] = v_target
            else:
                target_velocities[uav.uav_id] = np.zeros(3, dtype=float)

        # 3. Update physics & motion
        for uav in self.uavs:
            v_t = target_velocities.get(uav.uav_id, np.zeros(3, dtype=float))
            uav.update_motion(self.dt, v_t, self.goal_tolerance)

        # 4. Collision tracking between all pairs
        n_uavs = len(self.uavs)
        for i in range(n_uavs):
            u_i = self.uavs[i]
            if u_i.reached_goal or self.current_time < u_i.config.start_delay:
                continue
                
            min_dist = float("inf")
            for j in range(n_uavs):
                if i == j:
                    continue
                u_j = self.uavs[j]
                if self.current_time < u_j.config.start_delay:
                    continue
                d = float(np.linalg.norm(u_i.position - u_j.position))
                if d < min_dist:
                    min_dist = d
                    
            u_i.record_collision_step(self.dt, min_dist)

        # Advance clock
        self.current_time += self.dt
        self.step_count += 1

        # 5. Check termination condition
        all_reached = all(u.reached_goal for u in self.uavs)
        time_exceeded = self.current_time >= self.scenario.sim_time_limit
        
        if all_reached or time_exceeded:
            self.is_finished = True
            return False

        return True

    def run_all(self) -> Dict[str, Any]:
        """Runs the simulation to completion and returns metrics summary."""
        self.reset()
        while not self.is_finished:
            self.step()
        return calculate_swarm_summary(self.uavs, self.current_time)
