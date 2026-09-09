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
                sim_time_limit=30.0,
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
