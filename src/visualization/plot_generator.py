"""
Publication Quality Plotting and Paper Reproduction Charts.
Generates Trajectory Comparison plots, PTTR bar graphs, and cross-experiment summaries
matching figures from Min & Nam (Paper 1) and Adithya & Kundu (Paper 2).
"""
import os
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

class PaperPlotGenerator:
    """Generates comparison graphs and saves them as high-res PNG/PDF images."""
    
    @staticmethod
    def plot_paper1_trajectories(comparison_data: Dict[str, List[Any]], save_path: str = "results/plots/paper1_trajectories.png"):
        """
        Plots 2D top-down trajectories comparing APF, Adaptive APF, Dynamic APF, and ACACT
        matching Paper 1 Figures 8, 10, 11, 13.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        algos = list(comparison_data.keys())
        n_algos = len(algos)
        
        fig, axes = plt.subplots(1, n_algos, figsize=(4.5 * n_algos, 4.5), sharex=True, sharey=True)
        if n_algos == 1:
            axes = [axes]
            
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (algo_name, uav_list) in enumerate(comparison_data.items()):
            ax = axes[idx]
            ax.set_facecolor('#0f141d')
            ax.grid(True, color='#2c3e50', linestyle='--', alpha=0.5)
            
            for u_idx, uav in enumerate(uav_list):
                traj = np.array(uav.trajectory)
                col = colors[u_idx % len(colors)]
                if len(traj) > 0:
                    ax.plot(traj[:, 0], traj[:, 1], color=col, linewidth=2.0, label=f"{uav.name}")
                    ax.plot(traj[0, 0], traj[0, 1], 'o', color=col, markersize=7)
                    ax.plot(uav.goal[0], uav.goal[1], '*', color='#f1c40f', markersize=10)
                    
            ax.set_title(f"({chr(97+idx)}) {algo_name}", color='#2c3e50', fontsize=12, fontweight='bold')
            ax.set_xlabel("X (m)")
            if idx == 0:
                ax.set_ylabel("Y (m)")
                
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PlotGenerator] Saved trajectory comparison: {save_path}")

    @staticmethod
    def plot_paper1_metrics_comparison(
        metrics_by_algo: Dict[str, Dict[str, float]],
        scenario_title: str = "Scenario 2",
        save_path: str = "results/plots/paper1_metrics_barchart.png"
    ):
        """
        Generates 3-panel bar chart: (a) Traveling Distance, (b) Traveling Time, (c) PTTR
        matching Paper 1 Figures 12 & 15.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        algos = list(metrics_by_algo.keys())
        
        distances = [metrics_by_algo[a]['mean_trajectory_length'] for a in algos]
        times = [metrics_by_algo[a]['mean_travel_time'] for a in algos]
        pttrs = [metrics_by_algo[a]['mean_pttr_paper1'] for a in algos]
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        bar_colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
        
        # (a) Distance
        axes[0].bar(algos, distances, color=bar_colors[:len(algos)], width=0.55)
        axes[0].set_title("(a) Average Traveling Distance (m)", fontsize=11, fontweight='bold')
        axes[0].set_ylabel("Distance (m)")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0].tick_params(axis='x', rotation=15)
        
        # (b) Time
        axes[1].bar(algos, times, color=bar_colors[:len(algos)], width=0.55)
        axes[1].set_title("(b) Average Traveling Time (s)", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("Time (s)")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1].tick_params(axis='x', rotation=15)
        
        # (c) PTTR
        axes[2].bar(algos, pttrs, color=bar_colors[:len(algos)], width=0.55)
        axes[2].set_title("(c) Average PTTR Metric", fontsize=11, fontweight='bold')
        axes[2].set_ylabel("PTTR")
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        axes[2].tick_params(axis='x', rotation=15)
        
        plt.suptitle(f"Paper 1 Benchmark Evaluation - {scenario_title}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PlotGenerator] Saved Paper 1 metrics bar charts: {save_path}")

    @staticmethod
    def plot_paper2_experiments_summary(
        exp_summaries: Dict[str, Dict[str, Any]],
        save_path: str = "results/plots/paper2_experiments_summary.png"
    ):
        """
        Generates Paper 2 Cross-Experiment Summary matching Figures 10 and 11 from Adithya & Kundu.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        exp_names = list(exp_summaries.keys())
        mean_pttrs = [exp_summaries[e]['mean_pttr_paper2'] for e in exp_names]
        mean_times = [exp_summaries[e]['mean_travel_time'] for e in exp_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # Panel 1: Travel Time
        colors = ['#27ae60', '#c0392b', '#d35400', '#2980b9']
        axes[0].bar(exp_names, mean_times, color=colors[:len(exp_names)], width=0.5)
        axes[0].set_title("Mean Swarm Travel Time (s)", fontweight='bold')
        axes[0].set_ylabel("Time (s)")
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Panel 2: PTTR
        axes[1].bar(exp_names, mean_pttrs, color=colors[:len(exp_names)], width=0.5)
        axes[1].set_title("Mean Swarm PTTR (Paper 2 Formulation)", fontweight='bold')
        axes[1].set_ylabel("PTTR")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle("Paper 2 (DCACS) Cross-Experiment Validation (E1 - E4)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PlotGenerator] Saved Paper 2 experiments summary: {save_path}")

    @staticmethod
    def plot_fps_sctp_comparison(
        comparison_results: Dict[str, Dict[str, Any]],
        save_path: str = "results/plots/fps_sctp_agra_comparison.png"
    ):
        """
        Generates 4-panel comparison between DCACS and FPS-SCTP / AGRA:
        (a) PTTR, (b) Travel Time, (c) Retransmissions / Stale Packets Filtered, (d) Bandwidth Overhead.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        protocols = list(comparison_results.keys())
        
        pttrs = [comparison_results[p]['pttr'] for p in protocols]
        times = [comparison_results[p]['travel_time'] for p in protocols]
        rtx_counts = [comparison_results[p].get('retransmissions', 0) for p in protocols]
        filtered_counts = [comparison_results[p].get('late_filtered', 0) for p in protocols]
        
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        colors = ['#e67e22', '#2980b9', '#27ae60', '#8e44ad']
        
        # (a) PTTR
        axes[0, 0].bar(protocols, pttrs, color=colors[:len(protocols)], width=0.5)
        axes[0, 0].set_title("(a) Swarm PTTR Score", fontweight='bold')
        axes[0, 0].set_ylabel("PTTR")
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # (b) Travel Time
        axes[0, 1].bar(protocols, times, color=colors[:len(protocols)], width=0.5)
        axes[0, 1].set_title("(b) Travel Time (s)", fontweight='bold')
        axes[0, 1].set_ylabel("Time (s)")
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # (c) Retransmissions
        axes[1, 0].bar(protocols, rtx_counts, color='#c0392b', width=0.5)
        axes[1, 0].set_title("(c) Total Retransmissions", fontweight='bold')
        axes[1, 0].set_ylabel("Packets")
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # (d) Late Messages Filtered (LMF)
        axes[1, 1].bar(protocols, filtered_counts, color='#16a085', width=0.5)
        axes[1, 1].set_title("(d) Late Stale Messages Filtered (LMF)", fontweight='bold')
        axes[1, 1].set_ylabel("Filtered Packets")
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle("AGRA / FPS-SCTP vs DCACS Performance Comparison", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PlotGenerator] Saved FPS-SCTP / AGRA comparison charts: {save_path}")

    @staticmethod
    def plot_large_scale_benchmarks(
        benchmark_data: List[Dict[str, Any]],
        save_path: str = "results/plots/large_scale_dcacs_vs_fps_sctp.png"
    ):
        """
        Plots comprehensive 6-panel scalability trends across swarm sizes (4 to 50 UAVs):
        1. PTTR vs Swarm Size
        2. Travel Time vs Swarm Size
        3. Packet Loss Rate (%) vs Swarm Size
        4. Total Retransmissions vs Swarm Size
        5. Total Bandwidth Consumed (KB) vs Swarm Size
        6. Total Battery / Energy (mAh / Joules) vs Swarm Size
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        scenarios = [d['scenario'] for d in benchmark_data]
        swarm_sizes = [d['swarm_size'] for d in benchmark_data]
        
        dcacs_pttr = [d['dcacs_pttr'] for d in benchmark_data]
        fps_pttr = [d['fps_pttr'] for d in benchmark_data]
        
        dcacs_time = [d['dcacs_travel_time'] for d in benchmark_data]
        fps_time = [d['fps_travel_time'] for d in benchmark_data]
        
        dcacs_loss = [d['dcacs_loss_rate'] for d in benchmark_data]
        fps_loss = [d['fps_loss_rate'] for d in benchmark_data]
        
        dcacs_rtx = [d['dcacs_retransmissions'] for d in benchmark_data]
        fps_rtx = [d['fps_retransmissions'] for d in benchmark_data]
        
        dcacs_bw = [d['dcacs_bandwidth_kb'] for d in benchmark_data]
        fps_bw = [d['fps_bandwidth_kb'] for d in benchmark_data]
        
        dcacs_batt = [d['dcacs_battery_mah'] for d in benchmark_data]
        fps_batt = [d['fps_battery_mah'] for d in benchmark_data]
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        x_indices = np.arange(len(scenarios))
        w = 0.35
        
        # 1. PTTR
        axes[0, 0].bar(x_indices - w/2, dcacs_pttr, w, label='DCACS (Baseline)', color='#e67e22')
        axes[0, 0].bar(x_indices + w/2, fps_pttr, w, label='AGRA (FPS-SCTP)', color='#2980b9')
        axes[0, 0].set_title('1. Swarm PTTR Score vs Swarm Size', fontweight='bold')
        axes[0, 0].set_xticks(x_indices)
        axes[0, 0].set_xticklabels([f"{s}\n({n}U)" for s, n in zip(scenarios, swarm_sizes)], fontsize=9)
        axes[0, 0].set_ylabel('PTTR')
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0, 0].legend()
        
        # 2. Travel Time
        axes[0, 1].plot(x_indices, dcacs_time, 'o--', label='DCACS', color='#e67e22', linewidth=2)
        axes[0, 1].plot(x_indices, fps_time, 's-', label='FPS-SCTP', color='#2980b9', linewidth=2)
        axes[0, 1].set_title('2. Mean Travel Time (s)', fontweight='bold')
        axes[0, 1].set_xticks(x_indices)
        axes[0, 1].set_xticklabels(scenarios)
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)
        axes[0, 1].legend()
        
        # 3. Packet Loss Rate (%)
        axes[1, 0].bar(x_indices - w/2, dcacs_loss, w, label='DCACS', color='#e74c3c')
        axes[1, 0].bar(x_indices + w/2, fps_loss, w, label='FPS-SCTP', color='#27ae60')
        axes[1, 0].set_title('3. Packet Loss Rate (PLR %)', fontweight='bold')
        axes[1, 0].set_xticks(x_indices)
        axes[1, 0].set_xticklabels(scenarios)
        axes[1, 0].set_ylabel('Loss (%)')
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1, 0].legend()
        
        # 4. Retransmissions
        axes[1, 1].bar(x_indices - w/2, dcacs_rtx, w, label='DCACS', color='#c0392b')
        axes[1, 1].bar(x_indices + w/2, fps_rtx, w, label='FPS-SCTP (Graded LMF)', color='#16a085')
        axes[1, 1].set_title('4. Total Retransmissions (Overhead)', fontweight='bold')
        axes[1, 1].set_xticks(x_indices)
        axes[1, 1].set_xticklabels(scenarios)
        axes[1, 1].set_ylabel('Packets')
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        axes[1, 1].legend()
        
        # 5. Bandwidth Consumed (KB)
        axes[2, 0].plot(x_indices, dcacs_bw, 'o--', label='DCACS', color='#e67e22', linewidth=2)
        axes[2, 0].plot(x_indices, fps_bw, 's-', label='FPS-SCTP', color='#8e44ad', linewidth=2)
        axes[2, 0].set_title('5. Total Swarm Bandwidth Consumed (KB)', fontweight='bold')
        axes[2, 0].set_xticks(x_indices)
        axes[2, 0].set_xticklabels([f"{s}\n({n}U)" for s, n in zip(scenarios, swarm_sizes)], fontsize=9)
        axes[2, 0].set_ylabel('Bandwidth (KB)')
        axes[2, 0].grid(True, linestyle='--', alpha=0.7)
        axes[2, 0].legend()
        
        # 6. Battery / Energy Consumed (mAh)
        axes[2, 1].plot(x_indices, dcacs_batt, 'o--', label='DCACS', color='#e67e22', linewidth=2)
        axes[2, 1].plot(x_indices, fps_batt, 's-', label='FPS-SCTP', color='#2ecc71', linewidth=2)
        axes[2, 1].set_title('6. Total Battery Consumed (mAh)', fontweight='bold')
        axes[2, 1].set_xticks(x_indices)
        axes[2, 1].set_xticklabels([f"{s}\n({n}U)" for s, n in zip(scenarios, swarm_sizes)], fontsize=9)
        axes[2, 1].set_ylabel('Battery (mAh)')
        axes[2, 1].grid(True, linestyle='--', alpha=0.7)
        axes[2, 1].legend()
        
        plt.suptitle('Comprehensive Protocol Benchmark: DCACS vs. AGRA (FPS-SCTP) Scalability (4 to 50 UAVs)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PlotGenerator] Saved Large-Scale Scalability Charts: {save_path}")
