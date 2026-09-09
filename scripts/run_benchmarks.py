#!/usr/bin/env python3
"""
Automated Multi-Scenario Benchmark and Reproduction Suite.
Executes all experiments from Paper 1 (Scenarios 1-3 & Controllers) and Paper 2 (E1-E4 & DCACS),
producing formatted summary tables and saving benchmark datasets.
"""
import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.avoidance.apf import APFController
from src.avoidance.adaptive_apf import AdaptiveAPFController
from src.avoidance.dynamic_apf import DynamicAPFController
from src.avoidance.acact import ACACTController
from src.communication.ideal import IdealChannel
from src.communication.dcacs import DCACSChannel
from src.scenarios.scenario_loader import ScenarioFactory
from src.simulation.engine import SimulationEngine
from src.metrics.logger import SimulationLogger
from src.visualization.plot_generator import PaperPlotGenerator

def run_paper2_experiments():
    print("\n" + "=" * 80)
    print(" EXECUTING PAPER 2 (DCACS) EXPERIMENTS: E1, E2, E3, E4 ")
    print("=" * 80)
    
    exp_ids = ["E1", "E2", "E3", "E4"]
    exp_summaries = {}
    
    for exp_id in exp_ids:
        scenario = ScenarioFactory.create_paper2_experiment(exp_id)
        controller = ACACTController(t_s_base=0.5)
        channel = DCACSChannel(delay_threshold=0.1, congestion_threshold=3, fast_loss_rate=0.2)
        
        engine = SimulationEngine(scenario, controller, channel)
        summary = engine.run_all()
        exp_summaries[exp_id] = summary
        
        SimulationLogger.print_experiment_results_table(summary, exp_name=f"Paper 2 - {scenario.name}")
        SimulationLogger.export_results_json(summary, f"results/json/paper2_{scenario.name}.json")

    # Generate Paper 2 Cross-Experiment Summary Chart
    PaperPlotGenerator.plot_paper2_experiments_summary(exp_summaries, "results/plots/paper2_experiments_summary.png")
    return exp_summaries

def run_paper1_all_scenarios():
    print("\n" + "=" * 80)
    print(" EXECUTING PAPER 1 (ACACT) BENCHMARKS ACROSS SCENARIOS 1, 2, 3 & FORMATION ")
    print("=" * 80)
    
    controllers = [
        APFController(),
        AdaptiveAPFController(),
        DynamicAPFController(),
        ACACTController(t_s_base=2.0)
    ]
    
    # 1. Scenario 1 (2 UAVs Head-on)
    print("\n>>> Scenario 1: 2-UAV Head-on Crossing <<<")
    s1_metrics = {}
    s1_trajs = {}
    for ctrl in controllers:
        scenario = ScenarioFactory.create_paper1_scenario1(v_max=3.0, k_pp=1.5, t_s=2.0)
        engine = SimulationEngine(scenario, ctrl, IdealChannel())
        summary = engine.run_all()
        s1_metrics[ctrl.name] = summary
        s1_trajs[ctrl.name] = scenario.uavs
        SimulationLogger.print_experiment_results_table(summary, exp_name=f"Paper 1 - Scenario 1 ({ctrl.name})")
        SimulationLogger.export_results_json(summary, f"results/json/paper1_s1_{ctrl.name}.json")
    PaperPlotGenerator.plot_paper1_trajectories(s1_trajs, "results/plots/paper1_s1_trajectories.png")
    PaperPlotGenerator.plot_paper1_metrics_comparison(s1_metrics, scenario_title="Scenario 1 (2-UAV Head-On)", save_path="results/plots/paper1_s1_metrics_barchart.png")

    # 2. Scenario 2 (4 UAVs Diagonal)
    print("\n>>> Scenario 2: 4-UAV Diagonal Crossing <<<")
    s2_metrics = {}
    s2_trajs = {}
    for ctrl in controllers:
        scenario = ScenarioFactory.create_paper1_scenario2(v_max=3.0, k_pp=1.3, t_s=2.0)
        engine = SimulationEngine(scenario, ctrl, IdealChannel())
        summary = engine.run_all()
        s2_metrics[ctrl.name] = summary
        s2_trajs[ctrl.name] = scenario.uavs
        SimulationLogger.print_experiment_results_table(summary, exp_name=f"Paper 1 - Scenario 2 ({ctrl.name})")
        SimulationLogger.export_results_json(summary, f"results/json/paper1_s2_{ctrl.name}.json")
    PaperPlotGenerator.plot_paper1_trajectories(s2_trajs, "results/plots/paper1_s2_trajectories.png")
    PaperPlotGenerator.plot_paper1_metrics_comparison(s2_metrics, scenario_title="Scenario 2 (4-UAV Diagonal Crossing)", save_path="results/plots/paper1_s2_metrics_barchart.png")

    # 3. Scenario 3 (5 UAVs Multi-direction Charging)
    print("\n>>> Scenario 3: 5-UAV Intermittent Charging <<<")
    s3_metrics = {}
    s3_trajs = {}
    for ctrl in controllers:
        scenario = ScenarioFactory.create_paper1_scenario3(v_max=3.0, k_pp=1.3, t_s=2.0)
        engine = SimulationEngine(scenario, ctrl, IdealChannel())
        summary = engine.run_all()
        s3_metrics[ctrl.name] = summary
        s3_trajs[ctrl.name] = scenario.uavs
        SimulationLogger.print_experiment_results_table(summary, exp_name=f"Paper 1 - Scenario 3 ({ctrl.name})")
        SimulationLogger.export_results_json(summary, f"results/json/paper1_s3_{ctrl.name}.json")
    PaperPlotGenerator.plot_paper1_trajectories(s3_trajs, "results/plots/paper1_s3_trajectories.png")
    PaperPlotGenerator.plot_paper1_metrics_comparison(s3_metrics, scenario_title="Scenario 3 (5-UAV Intermittent Charging)", save_path="results/plots/paper1_s3_metrics_barchart.png")

    # 4. Formation Flight (10 Iris UAVs)
    print("\n>>> Formation Flight: 10 Iris UAVs Rotating <<<")
    form_scenario = ScenarioFactory.create_paper1_formation()
    form_engine = SimulationEngine(form_scenario, ACACTController(t_s_base=2.0), IdealChannel())
    form_summary = form_engine.run_all()
    SimulationLogger.print_experiment_results_table(form_summary, exp_name="Paper 1 - 10-UAV Formation Flight (ACACT)")
    SimulationLogger.export_results_json(form_summary, "results/json/paper1_formation_acact.json")
    PaperPlotGenerator.plot_paper1_trajectories({"ACACT_Formation": form_scenario.uavs}, "results/plots/paper1_formation_trajectories.png")

def main():
    print("\n================================================================================")
    print(" >>> FULL BENCHMARK SUITE: PAPER 1 (ACACT) & PAPER 2 (DCACS) REPRODUCTION <<< ")
    print("================================================================================\n")
    p2_results = run_paper2_experiments()
    p1_results = run_paper1_all_scenarios()
    print("\n" + "=" * 80)
    print(" ALL BENCHMARKS COMPLETED!")
    print(" Tables printed to stdout.")
    print(" JSON summaries saved in: results/json/")
    print(" Plots saved in:          results/plots/")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
