#!/usr/bin/env python3
"""
Large-Scale Multi-Scenario Protocol Comparison Suite: DCACS vs. AGRA (FPS-SCTP).
Executes Scenarios a1 through a10 (4 up to 50 UAVs) across multiple Monte Carlo runs,
evaluates all 7 core metric dimensions, prints formatted comparison tables,
and exports publication-quality CSV/JSON and figures.
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.avoidance.acact import ACACTController
from src.communication.dcacs import DCACSChannel
from src.communication.fps_sctp import FPSSCTPChannel
from src.scenarios.scenario_loader import ScenarioFactory
from src.simulation.engine import SimulationEngine
from src.metrics.logger import SimulationLogger
from src.visualization.plot_generator import PaperPlotGenerator

def run_multi_run_evaluation(scenario_id: str, channel_type: str, runs: int = 5) -> dict:
    """
    Executes multiple Monte Carlo simulation runs for a given scenario & protocol
    to compute statistical mean values.
    """
    pttrs = []
    times = []
    col_times = []
    loss_rates = []
    retransmissions = []
    bandwidths_kb = []
    energies_joules = []
    batteries_mah = []
    powers_watts = []
    lmf_filtered = []
    swarm_size = 0
    
    for r in range(runs):
        scenario = ScenarioFactory.create_scalable_scenario(scenario_id)
        swarm_size = len(scenario.uavs)
        controller = ACACTController(t_s_base=1.5)
        
        if channel_type.upper() == "DCACS":
            channel = DCACSChannel(delay_threshold=0.1, congestion_threshold=3, fast_loss_rate=0.2)
        else: # FPS-SCTP / AGRA
            channel = FPSSCTPChannel(base_rtt=0.04, channel_loss_rate=0.15, enable_lmf=True)
            
        engine = SimulationEngine(scenario, controller, channel)
        summary = engine.run_all()
        
        pttrs.append(summary['mean_pttr_paper2'])
        times.append(summary['mean_travel_time'])
        col_times.append(summary['mean_collision_time'])
        loss_rates.append(summary.get('packet_loss_rate_pct', 0.0))
        retransmissions.append(summary.get('total_retransmissions', 0))
        bandwidths_kb.append(summary.get('total_bandwidth_kb', 0.0))
        energies_joules.append(summary.get('total_energy_joules', 0.0))
        batteries_mah.append(summary.get('total_battery_mah', 0.0))
        powers_watts.append(summary.get('mean_power_watts', 0.0))
        
        if channel_type.upper() != "DCACS":
            stats = channel.get_stats()
            lmf_filtered.append(stats.get('late_messages_filtered', 0))
        else:
            lmf_filtered.append(0)

    return {
        "scenario": scenario_id.upper(),
        "swarm_size": swarm_size,
        "mean_pttr": float(np.mean(pttrs)),
        "std_pttr": float(np.std(pttrs)),
        "mean_time": float(np.mean(times)),
        "mean_col_time": float(np.mean(col_times)),
        "mean_loss_rate": float(np.mean(loss_rates)),
        "mean_rtx": float(np.mean(retransmissions)),
        "mean_bandwidth_kb": float(np.mean(bandwidths_kb)),
        "mean_energy_joules": float(np.mean(energies_joules)),
        "mean_battery_mah": float(np.mean(batteries_mah)),
        "mean_power_watts": float(np.mean(powers_watts)),
        "mean_lmf_filtered": float(np.mean(lmf_filtered))
    }

def main():
    parser = argparse.ArgumentParser(description="Multi-Scenario Comparison: DCACS vs AGRA (FPS-SCTP)")
    parser.add_argument("--scenarios", "-s", nargs="+", default=["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"], help="Scenarios to evaluate")
    parser.add_argument("--runs", "-r", type=int, default=5, help="Number of Monte Carlo runs per scenario for statistical averaging")
    args = parser.parse_args()

    print("\n" + "=" * 125)
    print(f" LARGE-SCALE MULTI-UAV BENCHMARK: DCACS vs. AGRA (FPS-SCTP) | {args.runs} RUNS PER SCENARIO ")
    print("=" * 125)

    comparison_plot_data = []
    table_rows = []
    raw_results = []

    for sc_id in args.scenarios:
        print(f"--> Evaluating {sc_id.upper()} across {args.runs} runs (DCACS & FPS-SCTP)...", end="", flush=True)
        
        res_dcacs = run_multi_run_evaluation(sc_id, "DCACS", runs=args.runs)
        res_fps = run_multi_run_evaluation(sc_id, "FPS_SCTP", runs=args.runs)
        print(" [DONE]")

        # Metrics delta
        pttr_gain_pct = ((res_fps['mean_pttr'] - res_dcacs['mean_pttr']) / max(1e-4, res_dcacs['mean_pttr'])) * 100.0
        batt_saved_pct = ((res_dcacs['mean_battery_mah'] - res_fps['mean_battery_mah']) / max(1e-4, res_dcacs['mean_battery_mah'])) * 100.0
        bw_saved_pct = ((res_dcacs['mean_bandwidth_kb'] - res_fps['mean_bandwidth_kb']) / max(1e-4, res_dcacs['mean_bandwidth_kb'])) * 100.0

        table_rows.append([
            f"{sc_id.upper()} ({res_fps['swarm_size']}U)",
            f"{res_dcacs['mean_pttr']:.4f}",
            f"{res_fps['mean_pttr']:.4f} (+{pttr_gain_pct:.1f}%)",
            f"{res_dcacs['mean_time']:.1f}s / {res_fps['mean_time']:.1f}s",
            f"{res_dcacs['mean_loss_rate']:.1f}% / {res_fps['mean_loss_rate']:.1f}%",
            f"{int(res_dcacs['mean_rtx'])} / {int(res_fps['mean_rtx'])}",
            f"{int(res_fps['mean_lmf_filtered'])} pkts",
            f"{res_dcacs['mean_bandwidth_kb']:.1f} / {res_fps['mean_bandwidth_kb']:.1f} KB",
            f"{res_dcacs['mean_battery_mah']:.1f} / {res_fps['mean_battery_mah']:.1f} mAh",
            f"{batt_saved_pct:+.1f}%"
        ])

        comparison_plot_data.append({
            "scenario": sc_id.upper(),
            "swarm_size": res_fps['swarm_size'],
            "dcacs_pttr": res_dcacs['mean_pttr'],
            "fps_pttr": res_fps['mean_pttr'],
            "dcacs_travel_time": res_dcacs['mean_time'],
            "fps_travel_time": res_fps['mean_time'],
            "dcacs_loss_rate": res_dcacs['mean_loss_rate'],
            "fps_loss_rate": res_fps['mean_loss_rate'],
            "dcacs_retransmissions": res_dcacs['mean_rtx'],
            "fps_retransmissions": res_fps['mean_rtx'],
            "dcacs_bandwidth_kb": res_dcacs['mean_bandwidth_kb'],
            "fps_bandwidth_kb": res_fps['mean_bandwidth_kb'],
            "dcacs_battery_mah": res_dcacs['mean_battery_mah'],
            "fps_battery_mah": res_fps['mean_battery_mah']
        })

        raw_results.append({
            "scenario": sc_id.upper(),
            "swarm_size": res_fps['swarm_size'],
            "dcacs": res_dcacs,
            "fps_sctp": res_fps
        })

    # Print Side-by-Side Comparison Table
    headers = [
        "Scenario", "PTTR (DCACS)", "PTTR (FPS-SCTP)",
        "Time (DC/FPS)", "Loss Rate", "Rtx (DC/FPS)",
        "LMF Filtered", "Bandwidth Used", "Battery (mAh)", "Energy Saved"
    ]
    print("\n" + tabulate(table_rows, headers=headers, tablefmt="github"))
    print("=" * 125 + "\n")

    # Export to CSV & JSON
    os.makedirs("results/csv", exist_ok=True)
    os.makedirs("results/json", exist_ok=True)
    
    df = pd.DataFrame(comparison_plot_data)
    df.to_csv("results/csv/large_scale_comparison_a1_a10.csv", index=False)
    
    import json
    with open("results/json/large_scale_comparison_a1_a10.json", "w") as f:
        json.dump(raw_results, f, indent=2)

    # Generate 6-panel scalability figure
    PaperPlotGenerator.plot_large_scale_benchmarks(comparison_plot_data, "results/plots/large_scale_dcacs_vs_fps_sctp.png")
    print(">> Benchmark results saved:")
    print("   - CSV Table:  results/csv/large_scale_comparison_a1_a10.csv")
    print("   - JSON Data:  results/json/large_scale_comparison_a1_a10.json")
    print("   - Scalability Chart: results/plots/large_scale_dcacs_vs_fps_sctp.png\n")

if __name__ == '__main__':
    main()
