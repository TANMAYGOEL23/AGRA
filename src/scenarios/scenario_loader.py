"""
Scenario Loader and Benchmark Definitions for Paper 1 (ACACT) and Paper 2 (DCACS).
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from src.core.uav import UAV, UAVConfig

@dataclass
class Scenario:
    name: str
    description: str
    uavs: List[UAV]
    bounds: List[float]  # [xmin, xmax, ymin, ymax, zmin, zmax]
    sim_time_limit: float = 60.0
    dt: float = 0.05
    is_3d: bool = True

class ScenarioFactory:
    """Creates fully configured scenarios for experiments."""
    
    @staticmethod
    def create_paper1_scenario1(v_max: float = 3.0, k_pp: float = 1.5, t_s: float = 2.0) -> Scenario:
        """Paper 1 Scenario 1: Two UAVs in 3D head-on trajectory crossing."""
        cfg1 = UAVConfig(uav_id=1, name="UAV_1", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=0.0)
        cfg2 = UAVConfig(uav_id=2, name="UAV_2", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=0.0)
        
        uav1 = UAV(cfg1, start_pos=np.array([0.0, 5.0, 5.0]), goal_pos=np.array([0.0, -5.0, 5.0]))
        uav2 = UAV(cfg2, start_pos=np.array([0.0, -5.0, 5.0]), goal_pos=np.array([0.0, 5.0, 5.0]))
        
        return Scenario(
            name="paper1_scenario1",
            description="Paper 1 Scenario 1: 2-UAV Head-on crossing A(0,5,5) <-> B(0,-5,5)",
            uavs=[uav1, uav2],
            bounds=[-8.0, 8.0, -8.0, 8.0, 0.0, 10.0],
            sim_time_limit=30.0,
            dt=0.05,
            is_3d=True
        )

    @staticmethod
    def create_paper1_scenario2(v_max: float = 3.0, k_pp: float = 1.3, t_s: float = 2.0) -> Scenario:
        """Paper 1 Scenario 2: 4 UAVs 3D 10x10m diagonal crossing."""
        starts = [
            np.array([5.0, 5.0, 5.0]),
            np.array([5.0, -5.0, 5.0]),
            np.array([-5.0, -5.0, 5.0]),
            np.array([-5.0, 5.0, 5.0])
        ]
        goals = [
            np.array([-5.0, -5.0, 5.0]), # A -> D
            np.array([-5.0, 5.0, 5.0]),  # B -> C
            np.array([5.0, 5.0, 5.0]),   # D -> A
            np.array([5.0, -5.0, 5.0])   # C -> B
        ]
        
        uavs = []
        for i in range(4):
            cfg = UAVConfig(uav_id=i+1, name=f"UAV_{i+1}", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=0.0)
            uavs.append(UAV(cfg, starts[i], goals[i]))
            
        return Scenario(
            name="paper1_scenario2",
            description="Paper 1 Scenario 2: 4-UAV 3D diagonal crossing at center (0,0,5)",
            uavs=uavs,
            bounds=[-10.0, 10.0, -10.0, 10.0, 0.0, 10.0],
            sim_time_limit=35.0,
            dt=0.05,
            is_3d=True
        )

    @staticmethod
    def create_paper1_scenario3(v_max: float = 3.0, k_pp: float = 1.3, t_s: float = 2.0) -> Scenario:
        """Paper 1 Scenario 3: 5 UAVs intermittent charging and straight-line traversal."""
        configs = [
            UAVConfig(uav_id=1, name="UAV_1", v_max=v_max, k_pp=k_pp, t_s=t_s, start_delay=0.0),
            UAVConfig(uav_id=2, name="UAV_2", v_max=v_max, k_pp=k_pp, t_s=t_s, start_delay=0.0),
            UAVConfig(uav_id=3, name="UAV_3", v_max=v_max, k_pp=k_pp, t_s=t_s, start_delay=0.0),
            UAVConfig(uav_id=4, name="UAV_4", v_max=v_max, k_pp=k_pp, t_s=t_s, start_delay=2.5),
            UAVConfig(uav_id=5, name="UAV_5", v_max=v_max, k_pp=k_pp, t_s=t_s, start_delay=2.5),
        ]
        starts = [
            np.array([-17.0, 0.0, 5.0]),
            np.array([-5.0, 5.0, 5.0]),
            np.array([-5.0, -5.0, 5.0]),
            np.array([5.0, 5.0, 5.0]),
            np.array([5.0, -5.0, 5.0])
        ]
        goals = [
            np.array([7.0, 0.0, 5.0]),
            np.array([-15.0, -5.0, 5.0]),
            np.array([-15.0, 5.0, 5.0]),
            np.array([-5.0, -5.0, 5.0]),
            np.array([-5.0, 5.0, 5.0])
        ]
        
        uavs = [UAV(configs[i], starts[i], goals[i]) for i in range(5)]
        return Scenario(
            name="paper1_scenario3",
            description="Paper 1 Scenario 3: 5-UAV intermittent charging / complex crossing",
            uavs=uavs,
            bounds=[-20.0, 10.0, -10.0, 10.0, 0.0, 10.0],
            sim_time_limit=40.0,
            dt=0.05,
            is_3d=True
        )

    @staticmethod
    def create_paper1_formation(num_followers: int = 9, radius: float = 6.0) -> Scenario:
        """Paper 1 Formation Flight: 10 Iris UAVs rotating counterclockwise around leader."""
        uavs = []
        # Leader / Reference drone at (0, 0, 5)
        leader_cfg = UAVConfig(uav_id=0, name="Leader", v_max=2.0)
        uavs.append(UAV(leader_cfg, np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 5.0])))
        
        # 9 Followers on circle
        angles = np.linspace(0, 2*np.pi, num_followers, endpoint=False)
        for i, ang in enumerate(angles):
            cfg = UAVConfig(uav_id=i+1, name=f"Follower_{i+1}", v_max=3.0, k_pp=1.5, t_s=2.0)
            x = radius * np.cos(ang)
            y = radius * np.sin(ang)
            # Target rotated counter-clockwise by 120 degrees
            target_x = radius * np.cos(ang + 2*np.pi/3)
            target_y = radius * np.sin(ang + 2*np.pi/3)
            uavs.append(UAV(cfg, np.array([x, y, 5.0]), np.array([target_x, target_y, 5.0])))
            
        return Scenario(
            name="paper1_formation_flight",
            description="Paper 1 Formation Flight: 10-UAV counterclockwise circular formation",
            uavs=uavs,
            bounds=[-12.0, 12.0, -12.0, 12.0, 0.0, 10.0],
            sim_time_limit=45.0,
            dt=0.05,
            is_3d=True
        )

    @staticmethod
    def create_paper2_experiment(exp_id: str) -> Scenario:
        """
        Paper 2 (Adithya & Kundu) Experiments:
        - E1: Baseline FAST (4 UAVs, all delays 0.05s)
        - E2: Baseline RELIABLE (4 UAVs, all delays 0.20s)
        - E3: Mixed-Delay Adaptive (4 UAVs, delays [0.05, 0.20, 0.05, 0.20])
        - E4: Congestion-Aware 5-UAV (delays [0.05, 0.20, 0.05, 0.20, 0.15])
        """
        starts_4 = [
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 0.0]),
            np.array([0.0, 10.0, 0.0])
        ]
        goals_4 = [
            np.array([10.0, 10.0, 0.0]),
            np.array([0.0, 10.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0])
        ]
        
        if exp_id == "E1":
            delays = [0.05, 0.05, 0.05, 0.05]
            uavs = [
                UAV(UAVConfig(uav_id=i+1, name=f"UAV{i+1}", v_max=1.0, k_pa=1.0, k_pp=2.0, t_s=0.5, network_delay=delays[i]),
                    starts_4[i], goals_4[i]) for i in range(4)
            ]
            return Scenario(
                name="E1_Baseline_FAST",
                description="Paper 2 E1: Baseline FAST (4 UAVs, delays=0.05s)",
                uavs=uavs,
                bounds=[-2.0, 12.0, -2.0, 12.0, 0.0, 2.0],
                sim_time_limit=45.0,
                dt=0.1,  # 10Hz control loop as in Paper 2
                is_3d=False
            )
        elif exp_id == "E2":
            delays = [0.20, 0.20, 0.20, 0.20]
            uavs = [
                UAV(UAVConfig(uav_id=i+1, name=f"UAV{i+1}", v_max=1.0, k_pa=1.0, k_pp=2.0, t_s=0.5, network_delay=delays[i]),
                    starts_4[i], goals_4[i]) for i in range(4)
            ]
            return Scenario(
                name="E2_Baseline_RELIABLE",
                description="Paper 2 E2: Baseline RELIABLE (4 UAVs, delays=0.20s)",
                uavs=uavs,
                bounds=[-2.0, 12.0, -2.0, 12.0, 0.0, 2.0],
                sim_time_limit=60.0,
                dt=0.1,
                is_3d=False
            )
        elif exp_id == "E3":
            delays = [0.05, 0.20, 0.05, 0.20]
            uavs = [
                UAV(UAVConfig(uav_id=i+1, name=f"UAV{i+1}", v_max=1.0, k_pa=1.0, k_pp=2.0, t_s=0.5, network_delay=delays[i]),
                    starts_4[i], goals_4[i]) for i in range(4)
            ]
            return Scenario(
                name="E3_Mixed_Delay_Adaptive",
                description="Paper 2 E3: Mixed-Delay Adaptive (4 UAVs, delays=[0.05, 0.20, 0.05, 0.20])",
                uavs=uavs,
                bounds=[-2.0, 12.0, -2.0, 12.0, 0.0, 2.0],
                sim_time_limit=60.0,
                dt=0.1,
                is_3d=False
            )
        elif exp_id == "E4":
            delays = [0.05, 0.20, 0.05, 0.20, 0.15]
            starts_5 = starts_4 + [np.array([5.0, -2.0, 0.0])]
            goals_5 = goals_4 + [np.array([5.0, 12.0, 0.0])]
            uavs = [
                UAV(UAVConfig(uav_id=i+1, name=f"UAV{i+1}", v_max=1.0, k_pa=1.0, k_pp=2.0, t_s=0.5, network_delay=delays[i]),
                    starts_5[i], goals_5[i]) for i in range(5)
            ]
            return Scenario(
                name="E4_Congestion_Aware_5UAV",
                description="Paper 2 E4: Congestion-Aware (5 UAVs, vertical crosser, mixed delays)",
                uavs=uavs,
                bounds=[-2.0, 12.0, -3.0, 13.0, 0.0, 2.0],
                sim_time_limit=45.0,
                dt=0.1,
                is_3d=False
            )
        else:
            raise ValueError(f"Unknown experiment ID: {exp_id}")

    @staticmethod
    def create_scalable_scenario(scenario_id: str, v_max: float = 2.5, k_pp: float = 1.8, t_s: float = 1.5) -> Scenario:
        """
        Creates scalable swarm scenarios a1 through a10 supporting up to 50 UAVs.
        - a1 : 4 UAVs (Diagonal 10x10m crossing)
        - a2 : 8 UAVs (Circular Antipodal Inversion, R=12m)
        - a3 : 12 UAVs (Double Concentric Ring Inversion, R=8m & 14m)
        - a4 : 16 UAVs (Opposing Grid Flow, 8 vs 8)
        - a5 : 20 UAVs (3D Spherical Layer Crossing, R=15m)
        - a6 : 25 UAVs (Narrow Corridor Bottleneck Crossing)
        - a7 : 30 UAVs (Tri-Directional 3-Flock Swarm Crossing at 120 deg)
        - a8 : 36 UAVs (Triple Concentric Ring Inversion, 3x12)
        - a9 : 42 UAVs (Dense 3D Matrix Crossing)
        - a10: 50 UAVs (Large-Scale Dense Swarm Inversion, R=20m)
        """
        sc_id = scenario_id.lower().strip()
        
        # Determine drone count and pattern
        swarm_sizes = {
            "a1": 4, "a2": 8, "a3": 12, "a4": 16, "a5": 20,
            "a6": 25, "a7": 30, "a8": 36, "a9": 42, "a10": 50
        }
        
        if sc_id not in swarm_sizes:
            raise ValueError(f"Unknown scalable scenario ID '{scenario_id}'. Must be one of a1 .. a10.")
            
        n = swarm_sizes[sc_id]
        uavs = []
        bounds = [-25.0, 25.0, -25.0, 25.0, 0.0, 15.0]
        sim_time = 40.0 + (n * 0.4)
        
        if sc_id == "a1": # 4 UAVs
            return ScenarioFactory.create_paper2_experiment("E1")
            
        elif sc_id in ["a2", "a7", "a10"]: # Ring / Antipodal Swarm Inversion
            radius = 12.0 if sc_id == "a2" else (18.0 if sc_id == "a7" else 22.0)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            for i, ang in enumerate(angles):
                sx = radius * np.cos(ang)
                sy = radius * np.sin(ang)
                sz = 5.0 + (1.5 * np.sin(2 * ang) if n >= 20 else 0.0)
                
                # Antipodal target on opposite side of circle
                gx = radius * np.cos(ang + np.pi)
                gy = radius * np.sin(ang + np.pi)
                gz = sz
                
                # Heterogeneous delay distribution for realism
                delay = 0.05 if (i % 2 == 0) else 0.15
                cfg = UAVConfig(uav_id=i+1, name=f"UAV_{i+1}", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=delay)
                uavs.append(UAV(cfg, np.array([sx, sy, sz]), np.array([gx, gy, gz])))
                
            bounds = [-radius - 4, radius + 4, -radius - 4, radius + 4, 0.0, 12.0]
            
        elif sc_id in ["a3", "a8"]: # Concentric Multi-Ring Inversion
            num_rings = 2 if sc_id == "a3" else 3
            drones_per_ring = int(n / num_rings)
            radii = [8.0, 14.0] if num_rings == 2 else [8.0, 14.0, 20.0]
            
            u_count = 1
            for r_idx, r in enumerate(radii):
                angs = np.linspace(0, 2 * np.pi, drones_per_ring, endpoint=False) + (r_idx * 0.2)
                for ang in angs:
                    sx = r * np.cos(ang)
                    sy = r * np.sin(ang)
                    sz = 4.0 + (r_idx * 2.0)
                    gx = r * np.cos(ang + np.pi)
                    gy = r * np.sin(ang + np.pi)
                    gz = sz
                    delay = 0.05 + (0.05 * (u_count % 3))
                    cfg = UAVConfig(uav_id=u_count, name=f"UAV_{u_count}", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=delay)
                    uavs.append(UAV(cfg, np.array([sx, sy, sz]), np.array([gx, gy, gz])))
                    u_count += 1
            max_r = max(radii)
            bounds = [-max_r - 4, max_r + 4, -max_r - 4, max_r + 4, 0.0, 15.0]

        elif sc_id in ["a4", "a6"]: # Opposing Flow / Corridor
            half = int(n / 2)
            for i in range(half):
                y_offset = -12.0 + (i * (24.0 / max(1, half - 1)))
                z_offset = 5.0 + (1.0 * (i % 3))
                # Flow Left-to-Right
                cfg_lr = UAVConfig(uav_id=i+1, name=f"UAV_LR_{i+1}", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=0.05)
                uavs.append(UAV(cfg_lr, np.array([-15.0, y_offset, z_offset]), np.array([15.0, y_offset, z_offset])))
                
                # Flow Right-to-Left
                cfg_rl = UAVConfig(uav_id=half+i+1, name=f"UAV_RL_{i+1}", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=0.15)
                uavs.append(UAV(cfg_rl, np.array([15.0, y_offset + 0.5, z_offset]), np.array([-15.0, y_offset + 0.5, z_offset])))
            bounds = [-20.0, 20.0, -16.0, 16.0, 0.0, 12.0]

        elif sc_id in ["a5", "a9"]: # 3D Spherical Cloud Inversion
            for i in range(n):
                phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
                theta = np.pi * (1.0 + 5.0 ** 0.5) * i
                r = 16.0
                sx = r * np.sin(phi) * np.cos(theta)
                sy = r * np.sin(phi) * np.sin(theta)
                sz = 6.0 + (r * 0.4 * np.cos(phi))
                
                gx = -sx
                gy = -sy
                gz = 6.0 - (r * 0.4 * np.cos(phi))
                
                delay = 0.05 if (i % 3 == 0) else (0.15 if i % 3 == 1 else 0.20)
                cfg = UAVConfig(uav_id=i+1, name=f"UAV_{i+1}", v_max=v_max, k_pp=k_pp, t_s=t_s, network_delay=delay)
                uavs.append(UAV(cfg, np.array([sx, sy, sz]), np.array([gx, gy, gz])))
            bounds = [-22.0, 22.0, -22.0, 22.0, 0.0, 15.0]

        return Scenario(
            name=f"Scenario_{sc_id.upper()}_{n}UAVs",
            description=f"Scalable Swarm Benchmark {sc_id.upper()} with {n} UAVs",
            uavs=uavs,
            bounds=bounds,
            sim_time_limit=sim_time,
            dt=0.08,
            is_3d=True
        )
