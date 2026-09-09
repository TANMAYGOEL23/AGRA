# Multi-UAV Swarm Simulation Platform (ACACT + DCACS)

A modular, research-grade multi-UAV simulation platform integrating:
1. **Paper 1 (ACACT)**: *Adaptive Collision Avoidance Algorithm Based on Estimated Collision Time for Swarm UAVs* (Min & Nam, IEEE Access 2023).
2. **Paper 2 (DCACS)**: *Delay- and Congestion-Aware Adaptive Communication Switching for Distributed Multi-UAV Collision Avoidance Using ROS* (Adithya & Kundu, IEEE Access).

---

## Key Highlights & Modular Architecture

- **Pluggable Communication Layer (`BaseCommChannel`)**: Seamlessly experiment with different communication techniques and network dynamics (e.g. Ideal, DCACS, 802.11p/DSRC, 5G Sidelink/C-V2X, CSMA/CA, TDMA, and loss/jitter models) without modifying the collision avoidance controllers or UAV physics.
- **Pluggable Avoidance Controllers (`BaseAvoidanceController`)**:
  - `APFController` (Khatib 1986)
  - `AdaptiveAPFController` (Distance-adaptive damping)
  - `DynamicAPFController` (Relative-velocity normal force $\vec{r}_{vn}$)
  - `ACACTController` (3D projected target velocity, contingency plan local minima escape, oscillation cancellation, swarm aggregation)
- **Built-in Benchmark Scenarios**:
  - **Paper 1**: Scenario 1 (2 UAVs head-on), Scenario 2 (4 UAVs diagonal crossing), Scenario 3 (5 UAVs intermittent charging), Formation Flight (10 Iris UAVs).
  - **Paper 2**: E1 (Baseline FAST), E2 (Baseline RELIABLE), E3 (Mixed-Delay Adaptive), E4 (5-UAV Congestion-Aware).
- **Dual Runtime Support**:
  1. High-speed standalone Python/NumPy engine with real-time Pygame Forces & Telemetry visualizer (runs natively on macOS, Linux, Windows).
  2. Complete ROS Noetic / ROS2 distributed package (`swarm_acact`) with launch files and node scripts.

---

## Directory Structure

```
CN_simulation/
├── src/
│   ├── core/                        # 3D/2D Kinematics, UAV State, ENU/Local transforms
│   ├── avoidance/                   # APF, Adaptive APF, Dynamic APF, ACACT Controllers
│   ├── communication/               # BaseCommChannel, Ideal, DCACS, WirelessMesh (802.11p/5G)
│   ├── scenarios/                   # Scenario loader & presets for all paper experiments
│   ├── metrics/                     # PTTR (Paper 1 & 2), CTR, TTR, Flight Logger
│   ├── simulation/                  # Discrete-step multi-UAV simulation engine
│   ├── visualization/               # Real-time force visualizer (Green/Red/Blue) & Plot generator
│   └── ros_nodes/swarm_acact/       # Full ROS package (uav_node.py, launch files)
├── scripts/
│   ├── run_simulation.py            # Interactive runner with GUI & scenario selector
│   ├── run_benchmarks.py            # Automated reproduction suite for all paper experiments
│   └── plot_results.py              # Export publication-quality figures (PNG/PDF)
├── tests/                           # Unit & integration test suite
├── requirements.txt
└── README.md
```

---

## Installation

```bash
cd /Users/tanmaygoel/Documents/PROJECTS/CN_simulation
pip install -r requirements.txt
```

---

## How to Run

### 1. Interactive Simulation with Live Forces Visualizer

Run any scenario with real-time vector visualization (Green = Attractive, Red = Repulsive, Cyan = ACACT Target Velocity):

```bash
# Run Paper 2 Congestion Experiment (E4) with DCACS and ACACT:
python scripts/run_simulation.py --scenario e4 --controller acact --channel dcacs

# Run Paper 1 Scenario 2 with ACACT:
python scripts/run_simulation.py --scenario s2 --controller acact --channel ideal

# Run 10-UAV Formation Flight:
python scripts/run_simulation.py --scenario formation --controller acact

# Run Headless Mode (no GUI, high speed):
python scripts/run_simulation.py --scenario e4 --no-gui
```

### 2. Run Automated Benchmarks & Paper Reproduction

Run the full benchmark suite across all scenarios, algorithms, and experiments:

```bash
python scripts/run_benchmarks.py
```
This generates:
- Full terminal evaluation tables matching Tables 2–6 in Paper 2 and Figures 8, 12, 15 in Paper 1.
- High-resolution comparison plots saved to `results/plots/`.
- JSON data exports saved to `results/json/`.

### 3. Run Unit Tests

```bash
pytest tests/ -v
```

---

## Extending with Custom Communication Techniques

To implement a new communication protocol (e.g. 5G Sidelink, TDMA slotting, or custom packet routing):
1. Inherit from `BaseCommChannel` in `src/communication/channel.py`:
```python
from src.communication.channel import BaseCommChannel

class MyCustom5GChannel(BaseCommChannel):
    def __init__(self, latency_ms=10.0, reliability=0.99):
        super().__init__(name="5G_Sidelink")
        self.latency_s = latency_ms / 1000.0
        self.reliability = reliability

    def update_uav_communication(self, uav_list, current_time, dt):
        for src in uav_list:
            for dst in uav_list:
                if src.uav_id != dst.uav_id:
                    # Implement your custom channel model, SINR, packet queuing, etc.
                    dst.perceived_neighbors[src.uav_id] = {
                        'pos': src.position.copy(),
                        'vel': src.velocity.copy(),
                        'timestamp': current_time
                    }
```
2. Pass your channel to `SimulationEngine`:
```python
engine = SimulationEngine(scenario, controller, MyCustom5GChannel())
summary = engine.run_all()
```
