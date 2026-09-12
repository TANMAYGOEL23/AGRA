#!/usr/bin/env python3

import sys
import os
import argparse
import time
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.core.uav import UAVConfig, UAV
from src.avoidance.apf import APFController
from src.avoidance.adaptive_apf import AdaptiveAPFController
from src.avoidance.dynamic_apf import DynamicAPFController
from src.avoidance.acact import ACACTController
from src.communication.ideal import IdealChannel, FixedLossyChannel
from src.communication.dcacs import DCACSChannel
from src.communication.fps_sctp import FPSSCTPChannel
from src.communication.wireless_mesh import WirelessMeshChannel
from src.scenarios.scenario_loader import ScenarioFactory
from src.simulation.engine import SimulationEngine
from src.visualization.force_monitor import ForceMonitorVisualizer
from src.metrics.logger import SimulationLogger

def get_controller(name: str):
    name = name.lower()
    if name == "apf":
        return APFController()
    elif name == "adaptive_apf":
        return AdaptiveAPFController()
    elif name == "dynamic_apf":
        return DynamicAPFController()
    elif name == "acact":
        return ACACTController()
    else:
        raise ValueError(f"Unknown controller: {name}")

def get_channel(name: str):
    name = name.lower()
    if name == "ideal":
        return IdealChannel()
    elif name == "dcacs":
        return DCACSChannel()
    elif name in ["fps_sctp", "fps-sctp", "agra"]:
        return FPSSCTPChannel(base_rtt=0.04, channel_loss_rate=0.15, enable_lmf=True)
    elif name == "lossy":
        return FixedLossyChannel(loss_rate=0.2, delay=0.05)
    elif name in ["mesh", "802.11p"]:
        return WirelessMeshChannel(mac_protocol="CSMA_CA")
    else:
        raise ValueError(f"Unknown communication channel: {name}")

def get_scenario(name: str):
    name = name.lower().strip()
    if name in ["s1", "p1_s1", "paper1_scenario1"]:
        return ScenarioFactory.create_paper1_scenario1()
    elif name in ["s2", "p1_s2", "paper1_scenario2"]:
        return ScenarioFactory.create_paper1_scenario2()
    elif name in ["s3", "p1_s3", "paper1_scenario3"]:
        return ScenarioFactory.create_paper1_scenario3()
    elif name in ["formation", "p1_formation"]:
        return ScenarioFactory.create_paper1_formation()
    elif name in ["e1", "e1_fast"]:
        return ScenarioFactory.create_paper2_experiment("E1")
    elif name in ["e2", "e2_reliable"]:
        return ScenarioFactory.create_paper2_experiment("E2")
    elif name in ["e3", "e3_mixed"]:
        return ScenarioFactory.create_paper2_experiment("E3")
    elif name in ["e4", "e4_5uav"]:
        return ScenarioFactory.create_paper2_experiment("E4")
    elif name in ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]:
        return ScenarioFactory.create_scalable_scenario(name)
    else:
        raise ValueError(f"Unknown scenario name: {name}. Options: s1..s3, formation, e1..e4, a1..a10")

def main():
    parser = argparse.ArgumentParser(description="Multi-UAV Swarm Collision Avoidance & Communication Simulation")
    parser.add_argument("--scenario", "-s", type=str, default="e4", help="Scenario preset: s1..s3, formation, e1..e4, a1..a10 (up to 50 UAVs)")
    parser.add_argument("--controller", "-c", type=str, default="acact", help="Avoidance controller: apf, adaptive_apf, dynamic_apf, acact")
    parser.add_argument("--channel", "-n", type=str, default="dcacs", help="Communication channel: ideal, lossy, dcacs, fps_sctp (agra), mesh")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--fps", type=int, default=30, help="Visualization frames per second")
    args = parser.parse_args()

    scenario = get_scenario(args.scenario)
    controller = get_controller(args.controller)
    channel = get_channel(args.channel)

    print(f"\n========================================================")
    print(f" Starting Simulation: {scenario.name}")
    print(f" Controller : {controller.name}")
    print(f" Comm Layer : {channel.name}")
    print(f" UAV Count  : {len(scenario.uavs)}")
    print(f"========================================================\n")

    engine = SimulationEngine(scenario, controller, channel)
    engine.reset()

    visualizer = None
    if not args.no_gui:
        visualizer = ForceMonitorVisualizer(title=f"Multi-UAV Simulator - {scenario.name} ({controller.name})")
        visualizer.fps = args.fps
        if not visualizer.init_display():
            visualizer = None

    try:
        running = True
        while running and not engine.is_finished:
            if visualizer:
                if not visualizer.handle_events():
                    break
                if visualizer.paused:
                    time.sleep(0.05)
                    continue

            # Simulation Tick
            ongoing = engine.step()
            
            if visualizer:
                visualizer.render_frame(engine, scenario.bounds)
            else:
                # Print progress every 2 seconds
                if engine.step_count % int(2.0 / scenario.dt) == 0:
                    print(f"[t = {engine.current_time:.2f}s] Active UAVs navigating...")

            if not ongoing:
                break

    except KeyboardInterrupt:
        print("\n[Simulation interrupted by user]")
    finally:
        if visualizer:
            time.sleep(1.0)
            visualizer.close()

    # Calculate and print final metrics
    from src.metrics.pttr import calculate_swarm_summary
    summary = calculate_swarm_summary(engine.uavs, engine.current_time)
    SimulationLogger.print_experiment_results_table(summary, exp_name=f"{scenario.name} with {controller.name} + {channel.name}")
    
    # Save results JSON
    SimulationLogger.export_results_json(summary, f"results/json/{scenario.name}_{controller.name}.json")

if __name__ == '__main__':
    main()
