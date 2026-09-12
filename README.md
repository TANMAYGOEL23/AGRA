# Multi-UAV Swarm Simulation Platform (ACACT + DCACS + AGRA/FPS-SCTP)

A modular, research-grade multi-UAV simulation platform integrating:
1. **Paper 1 (ACACT)**: *Adaptive Collision Avoidance Algorithm Based on Estimated Collision Time for Swarm UAVs* (Min & Nam, IEEE Access 2023).
2. **Paper 2 (DCACS)**: *Delay- and Congestion-Aware Adaptive Communication Switching for Distributed Multi-UAV Collision Avoidance Using ROS* (Adithya & Kundu, IEEE Access).
3. **Research Direction (AGRA / FPS-SCTP)**: *Adaptive Graded Retransmission & Forward Prediction Scheduling SCTP for Swarm Collision Avoidance* (Inspired by Ronaldo et al., EMITTER 2022 and AGRA Team 14 Review).

---

## Key Highlights & Modular Architecture

- **Pluggable Communication Layer (`BaseCommChannel`)**: Seamlessly experiment with different communication techniques and network dynamics:
  - `IdealChannel`: Zero-latency, zero-loss theoretical baseline (Paper 1 assumption).
  - `DCACSChannel`: Delay- & Congestion-Aware binary protocol switching (FAST vs. RELIABLE) with delay threshold $\delta = 0.1\text{s}$ and congestion threshold $\gamma = 3$ (Paper 2).
  - `FPSSCTPChannel` (**AGRA**): Forward Prediction Scheduling SCTP with **Late Messages Filter (LMF)**, **Packet Delay Budget (PDB)**, and **Graded Retransmission Quotas** based on collision urgency $C_n$.
  - `FixedLossyChannel`: Configurable Bernoulli packet loss and delay channel.
  - `WirelessMeshChannel`: 802.11p/DSRC, 5G NR Sidelink, CSMA/CA MAC contention, and distance-based RF path-loss model.
- **Pluggable Avoidance Controllers (`BaseAvoidanceController`)**:
  - `APFController` (Khatib 1986)
  - `AdaptiveAPFController` (Distance-adaptive damping)
  - `DynamicAPFController` (Relative-velocity normal force $\vec{r}_{vn}$, Du et al. 2019)
  - `ACACTController` (3D projected target velocity $\vec{v}_v^+$, contingency escape plan, oscillation cancellation, swarm aggregation)
- **Built-in Benchmark Scenarios**:
  - **Paper 1**: Scenario 1 (2 UAVs head-on), Scenario 2 (4 UAVs diagonal crossing), Scenario 3 (5 UAVs intermittent charging), Formation Flight (10 Iris UAVs).
  - **Paper 2**: E1 (Baseline FAST), E2 (Baseline RELIABLE), E3 (Mixed-Delay Adaptive), E4 (5-UAV Congestion-Aware).
- **Dual Runtime Support**:
  1. High-speed standalone Python/NumPy engine with real-time Pygame Forces & Telemetry visualizer (runs natively on macOS, Linux, Windows).
  2. Complete ROS Noetic / ROS2 distributed package (`swarm_acact`) with launch files and node scripts.

---

## FPS-SCTP & AGRA: Adaptive Graded Retransmission

### Limitations of DCACS Addressed by AGRA / FPS-SCTP
- **Binary FAST / RELIABLE Switching**: DCACS only switches between two extremes (20% loss with 0 delay vs. 0% loss with 200ms delay), with no middle ground.
- **Blind Retransmission Overhead**: TCP-like RELIABLE mode retransmits every lost packet indiscriminately, causing queueing delays and network congestion.
- **Late Stale Data**: Retransmitting position packets that arrive after the collision has already occurred or moved wastes channel capacity.

### The FPS-SCTP & AGRA Solution
1. **Packet Delay Budget (PDB)**:
   $$\text{PDB} = \text{clip}\left(\frac{\|\mathbf{r}_{min}\|}{\|\mathbf{v}_i\|} \times 0.4, 0.08\text{ s}, 0.50\text{ s}\right)$$
   Derives a per-packet hard deadline from the UAV's current kinematics and distance to nearest obstacle/neighbor.
2. **Late Messages Filter (LMF)**:
   Before retransmitting a lost packet, forward-predict the retransmission arrival time $t_{current} + t_{Rtx}$.
   $$\text{Decision} = \begin{cases} \text{DROP (Late)}, & \text{if } t_{current} + t_{Rtx} > t_{creation} + \text{PDB} \\ \text{RETRANSMIT}, & \text{otherwise} \end{cases}$$
   Filters out stale messages to protect link capacity and maintain timeliness.
3. **Graded Urgency ($C_n$) & Multi-Streaming**:
   - $C_n \in [0.1, 1.0]$ based on proximity to collision zones.
   - High Urgency ($C_n > 0.75$): Allocated 2 retransmissions + routed to high-priority stream (Stream 1).
   - Moderate Urgency ($0.40 < C_n \le 0.75$): Allocated 1 retransmission.
   - Low Urgency ($C_n \le 0.40$): Best-effort (0 retransmissions, zero retransmission overhead).

---

## Directory Structure

```
CN_simulation/
├── configs/
│   ├── default_acact.yaml          # Paper 1 gains and physical bounds
│   ├── dcacs_config.yaml           # Paper 2 DCACS network thresholds
│   ├── fps_sctp_config.yaml        # FPS-SCTP & AGRA parameters (PDB, LMF, RTT)
│   └── formation_config.yaml       # 10-UAV formation configuration
├── src/
│   ├── core/                        # 3D/2D Kinematics, UAV State, ENU/Local transforms
│   ├── avoidance/                   # APF, Adaptive APF, Dynamic APF, ACACT Controllers
│   ├── communication/               # BaseCommChannel, Ideal, DCACS, FPS-SCTP (AGRA), WirelessMesh
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

## How to Run

### 1. Run Interactive Simulations with Real-Time Forces HUD

```bash
# Run 5-UAV Congestion Scenario (E4) with AGRA / FPS-SCTP:
python3 scripts/run_simulation.py --scenario e4 --controller acact --channel fps_sctp

# Run 4-UAV Mixed Delay Scenario (E3) with DCACS:
python3 scripts/run_simulation.py --scenario e3 --controller acact --channel dcacs

# Run Paper 1 Scenario 2 with ACACT:
python3 scripts/run_simulation.py --scenario s2 --controller acact --channel ideal

# Run 10-UAV Formation Flight:
python3 scripts/run_simulation.py --scenario formation --controller acact --channel ideal
```

### 2. Run Automated Benchmarks & Paper Reproduction

Run the full benchmark suite across all scenarios, algorithms, and experiments (including DCACS vs FPS-SCTP comparison):

```bash
python3 scripts/run_benchmarks.py
```
This automatically outputs:
- Summary tables for Paper 2 (E1–E4).
- Summary tables for Paper 1 (Scenarios 1, 2, 3, and 10-UAV Formation).
- Summary tables and telemetry for AGRA / FPS-SCTP vs DCACS.
- High-resolution comparison plots saved to `results/plots/`:
  - `fps_sctp_agra_comparison.png` (PTTR, Travel Time, Retransmissions, Late Filtered Packets)
  - `paper2_experiments_summary.png`
  - `paper1_s1_trajectories.png` & `paper1_s1_metrics_barchart.png`
  - `paper1_s2_trajectories.png` & `paper1_s2_metrics_barchart.png`
  - `paper1_s3_trajectories.png` & `paper1_s3_metrics_barchart.png`
  - `paper1_formation_trajectories.png`

### 3. Run Unit Tests

```bash
python3 -m unittest discover tests -v
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
