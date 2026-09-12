"""
Integration tests for Scenario Execution across all Paper 1 and Paper 2 Presets.
"""
import unittest
import numpy as np
from src.avoidance.acact import ACACTController
from src.avoidance.apf import APFController
from src.communication.ideal import IdealChannel
from src.communication.dcacs import DCACSChannel
from src.scenarios.scenario_loader import ScenarioFactory
from src.simulation.engine import SimulationEngine

class TestScenarios(unittest.TestCase):
    def test_paper1_scenario1_execution(self):
        scenario = ScenarioFactory.create_paper1_scenario1(v_max=3.0, k_pp=1.5, t_s=2.0)
        controller = ACACTController(t_s_base=2.0)
        channel = IdealChannel()
        engine = SimulationEngine(scenario, controller, channel)
        summary = engine.run_all()
        
        self.assertTrue(all(u.reached_goal for u in scenario.uavs))
        self.assertLess(summary['mean_travel_time'], 10.0)
        self.assertGreater(summary['mean_pttr_paper1'], 0.0)

    def test_paper1_scenario2_execution(self):
        scenario = ScenarioFactory.create_paper1_scenario2(v_max=3.0, k_pp=1.3, t_s=2.0)
        controller = ACACTController(t_s_base=2.0)
        channel = IdealChannel()
        engine = SimulationEngine(scenario, controller, channel)
        summary = engine.run_all()
        
        self.assertTrue(all(u.reached_goal for u in scenario.uavs))
        self.assertGreater(summary['mean_travel_time'], 0.0)

    def test_paper2_e1_fast_execution(self):
        scenario = ScenarioFactory.create_paper2_experiment("E1")
        controller = ACACTController(t_s_base=0.5)
        channel = DCACSChannel()
        engine = SimulationEngine(scenario, controller, channel)
        summary = engine.run_all()
        
        self.assertTrue(all(u.reached_goal for u in scenario.uavs))
        self.assertLess(summary['mean_travel_time'], 35.0)
        self.assertGreater(summary['mean_pttr_paper2'], 0.02)

    def test_paper2_e4_congestion_execution(self):
        scenario = ScenarioFactory.create_paper2_experiment("E4")
        controller = ACACTController(t_s_base=0.5)
        channel = DCACSChannel()
        engine = SimulationEngine(scenario, controller, channel)
        summary = engine.run_all()
        
        self.assertTrue(all(u.reached_goal for u in scenario.uavs))
        self.assertEqual(len(scenario.uavs), 5)

    def test_scalable_scenario_a2_energy_metrics(self):
        scenario = ScenarioFactory.create_scalable_scenario("a2")
        self.assertEqual(len(scenario.uavs), 8)
        controller = ACACTController(t_s_base=1.5)
        channel = IdealChannel()
        engine = SimulationEngine(scenario, controller, channel)
        summary = engine.run_all()
        
        self.assertGreater(summary['total_energy_joules'], 0.0)
        self.assertGreater(summary['total_battery_mah'], 0.0)
        self.assertGreater(summary['total_bandwidth_kb'], 0.0)

    def test_scalable_scenario_a10_initialization(self):
        scenario = ScenarioFactory.create_scalable_scenario("a10")
        self.assertEqual(len(scenario.uavs), 50)
        self.assertEqual(scenario.bounds[0], -26.0)

if __name__ == '__main__':
    unittest.main()
