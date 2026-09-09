"""
Simulation Logger, Table Formatter, and Metric Exporter.
"""
import json
import os
from typing import List, Dict, Any
from tabulate import tabulate
from src.metrics.pttr import UAVMetricsResult, calculate_swarm_summary
from src.core.uav import UAV

class SimulationLogger:
    """Logs and presents simulation results cleanly."""
    
    @staticmethod
    def print_experiment_results_table(summary: Dict[str, Any], exp_name: str = ""):
        print("\n" + "=" * 80)
        print(f" EXPERIMENT RESULTS: {exp_name.upper()} ")
        print("=" * 80)
        
        table_data = []
        for r in summary['per_uav']:
            table_data.append([
                r.name,
                f"{r.assigned_delay:.2f}s",
                r.final_comm_mode,
                f"{r.travel_time:.2f}s",
                f"{r.collision_time:.2f}s",
                f"{r.trajectory_length:.2f}m",
                f"{r.min_separation:.2f}m",
                f"{r.pttr_paper1:.4f}",
                f"{r.pttr_paper2:.5f}"
            ])
            
        headers = ["UAV", "Delay", "Mode", "Travel (s)", "Col (s)", "Path (m)", "Min Sep", "PTTR (P1)", "PTTR (P2)"]
        print(tabulate(table_data, headers=headers, tablefmt="github"))
        print("-" * 80)
        print(f"Mean Travel Time   : {summary['mean_travel_time']:.2f} s")
        print(f"Mean Collision Time: {summary['mean_collision_time']:.2f} s | Max: {summary['max_collision_time']:.2f} s")
        print(f"Mean Path Length   : {summary['mean_trajectory_length']:.2f} m")
        print(f"Mean PTTR (Paper 1): {summary['mean_pttr_paper1']:.4f}")
        print(f"Mean PTTR (Paper 2): {summary['mean_pttr_paper2']:.5f}")
        print("=" * 80 + "\n")

    @staticmethod
    def export_results_json(summary: Dict[str, Any], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Convert dataclasses to dict
        serializable = {
            "mean_travel_time": summary['mean_travel_time'],
            "mean_collision_time": summary['mean_collision_time'],
            "max_collision_time": summary['max_collision_time'],
            "mean_trajectory_length": summary['mean_trajectory_length'],
            "mean_pttr_paper1": summary['mean_pttr_paper1'],
            "mean_pttr_paper2": summary['mean_pttr_paper2'],
            "per_uav": [r.__dict__ for r in summary['per_uav']]
        }
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
