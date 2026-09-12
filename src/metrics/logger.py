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
        print("\n" + "=" * 105)
        print(f" EXPERIMENT RESULTS: {exp_name.upper()} (Swarm Size: {summary.get('swarm_size', len(summary['per_uav']))} UAVs)")
        print("=" * 105)
        
        table_data = []
        for r in summary['per_uav']:
            table_data.append([
                r.name,
                f"{r.travel_time:.2f}s",
                f"{r.collision_time:.2f}s",
                f"{r.pttr_paper2:.5f}",
                f"{r.packet_loss_rate_pct:.1f}%",
                f"{r.retransmissions}",
                f"{r.bandwidth_used_kb:.1f}KB",
                f"{r.energy_joules:.1f}J",
                f"{r.battery_mah:.1f}mAh",
                f"{r.avg_power_watts:.1f}W",
                r.final_comm_mode
            ])
            
        headers = ["UAV", "Travel(s)", "Col(s)", "PTTR", "Loss(%)", "Rtx", "Bandwidth", "Energy(J)", "Battery", "Power(W)", "Mode"]
        print(tabulate(table_data, headers=headers, tablefmt="github"))
        print("-" * 105)
        print(f"Mean Travel Time       : {summary['mean_travel_time']:.2f} s")
        print(f"Mean Collision Exposure: {summary['mean_collision_time']:.2f} s | Max: {summary['max_collision_time']:.2f} s")
        print(f"Mean PTTR Score        : {summary['mean_pttr_paper2']:.5f} (Paper 2) | {summary['mean_pttr_paper1']:.4f} (Paper 1)")
        print(f"Packet Loss Rate (PLR) : {summary.get('packet_loss_rate_pct', 0.0):.2f}%")
        print(f"Total Retransmissions  : {summary.get('total_retransmissions', 0)} (Overhead: {summary.get('comm_overhead_pct', 0.0):.1f}%)")
        print(f"Total Bandwidth Used   : {summary.get('total_bandwidth_kb', 0.0):.2f} KB")
        print(f"Swarm Battery Consumed : {summary.get('total_battery_mah', 0.0):.2f} mAh ({summary.get('total_energy_joules', 0.0):.1f} Joules | Mean Power: {summary.get('mean_power_watts', 0.0):.1f} W)")
        print("=" * 105 + "\n")

    @staticmethod
    def export_results_json(summary: Dict[str, Any], filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        serializable = {
            "swarm_size": summary.get("swarm_size", len(summary['per_uav'])),
            "mean_travel_time": summary['mean_travel_time'],
            "mean_collision_time": summary['mean_collision_time'],
            "max_collision_time": summary['max_collision_time'],
            "mean_trajectory_length": summary['mean_trajectory_length'],
            "mean_pttr_paper1": summary['mean_pttr_paper1'],
            "mean_pttr_paper2": summary['mean_pttr_paper2'],
            "total_energy_joules": summary.get('total_energy_joules', 0.0),
            "total_battery_mah": summary.get('total_battery_mah', 0.0),
            "mean_power_watts": summary.get('mean_power_watts', 0.0),
            "total_packets_sent": summary.get('total_packets_sent', 0),
            "total_packets_lost": summary.get('total_packets_lost', 0),
            "total_retransmissions": summary.get('total_retransmissions', 0),
            "packet_loss_rate_pct": summary.get('packet_loss_rate_pct', 0.0),
            "total_bandwidth_kb": summary.get('total_bandwidth_kb', 0.0),
            "comm_overhead_pct": summary.get('comm_overhead_pct', 0.0),
            "per_uav": [r.__dict__ for r in summary['per_uav']]
        }
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
