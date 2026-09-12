"""
UAV Physical Flight Dynamics and Communication Energy/Battery Consumption Model.
Models aerodynamic propulsion power (hover, velocity drag, acceleration),
wireless transceiver energy (TX, RX, Idle listening), and avionics computing power.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class DroneEnergySpecs:
    mass_kg: float = 1.5             # Quadrotor mass (e.g. 3DR Iris from Paper 1)
    gravity: float = 9.81            # m/s^2
    num_rotors: int = 4
    rotor_radius_m: float = 0.127    # 10-inch prop radius
    air_density: float = 1.225       # kg/m^3 at sea level
    drag_coefficient: float = 0.3    # C_D aerodynamic drag
    frontal_area_m2: float = 0.05    # Frontal projection area
    motor_efficiency: float = 0.80   # Electrical to mechanical conversion
    battery_voltage: float = 14.8    # 4S LiPo battery nominal voltage (V)
    
    # Wireless & Computing Electronics
    tx_power_w: float = 1.20         # 20 dBm RF output + PA power draw
    rx_power_w: float = 0.45         # Receiver listening draw
    idle_power_w: float = 0.15       # Radio baseband idle draw
    avionics_power_w: float = 3.50   # PX4 flight controller & companion computer

class UAVEnergyModel:
    """
    Computes instantaneous and accumulated energy expenditure for a UAV over time.
    """
    def __init__(self, specs: DroneEnergySpecs = None):
        self.specs = specs or DroneEnergySpecs()
        
        # Disk area for all rotors
        self.rotor_disk_area = self.specs.num_rotors * (np.pi * (self.specs.rotor_radius_m ** 2))
        # Induced hover velocity v_h
        thrust_hover = self.specs.mass_kg * self.specs.gravity
        self.v_hover = np.sqrt(thrust_hover / (2 * self.specs.air_density * self.rotor_disk_area))
        # Base mechanical hover power
        self.p_hover_base = (thrust_hover ** 1.5) / np.sqrt(2 * self.specs.air_density * self.rotor_disk_area)
        self.p_hover_electrical = self.p_hover_base / self.specs.motor_efficiency

    def compute_propulsion_power(self, speed: float, acceleration_mag: float = 0.0) -> float:
        """
        Calculates instantaneous aerodynamic propulsion power in Watts as a function of speed and acceleration.
        P_prop(v) = P_induced(v) + P_parasitic_drag(v) + P_inertial_acceleration
        """
        v = max(0.0, speed)
        
        # Induced power at forward speed v (momentum theory)
        # Ratio = sqrt( (-v^2 + sqrt(v^4 + 4*v_h^4)) / 2 ) / v_h
        term1 = - (v ** 2) + np.sqrt((v ** 4) + 4 * (self.v_hover ** 4))
        induced_ratio = np.sqrt(max(0.0, term1 / 2.0)) / max(1e-3, self.v_hover)
        p_induced = self.p_hover_electrical * induced_ratio
        
        # Parasitic drag power: 0.5 * rho * C_D * A * v^3
        p_drag = 0.5 * self.specs.air_density * self.specs.drag_coefficient * self.specs.frontal_area_m2 * (v ** 3)
        
        # Inertial acceleration power: m * a * v
        p_acc = self.specs.mass_kg * acceleration_mag * v
        
        total_p_prop = p_induced + p_drag + p_acc
        return float(np.clip(total_p_prop, 80.0, 450.0))

    def compute_comm_power(self, is_transmitting: bool, is_receiving: bool) -> float:
        """Calculates instantaneous transceiver power draw."""
        p_comm = self.specs.idle_power_w
        if is_transmitting:
            p_comm += self.specs.tx_power_w
        if is_receiving:
            p_comm += self.specs.rx_power_w
        return p_comm

    def calculate_mission_energy(
        self,
        trajectory: List[np.ndarray],
        velocity_history: List[np.ndarray],
        total_time: float,
        dt: float,
        packets_sent: int,
        packets_recv: int,
        retransmissions: int = 0
    ) -> Dict[str, float]:
        """
        Calculates integrated mission energy in Joules, Watt-Hours (Wh), and mAh consumed.
        """
        if len(velocity_history) < 2:
            avg_speed = 1.0
            speeds = [avg_speed]
            accels = [0.0]
        else:
            vels = np.array(velocity_history)
            speeds = np.linalg.norm(vels, axis=1)
            # Differentiate speed to get acceleration
            accels = np.abs(np.diff(speeds, prepend=speeds[0])) / max(1e-4, dt)

        # 1. Integrate propulsion energy
        propulsion_energy_joules = 0.0
        for s, a in zip(speeds, accels):
            p_prop = self.compute_propulsion_power(s, a)
            propulsion_energy_joules += p_prop * dt

        # 2. Integrate electronics & computation energy
        avionics_energy_joules = self.specs.avionics_power_w * total_time

        # 3. Integrate communication energy
        # Assume packet airtime approx 0.5ms per packet transmission
        tx_airtime_s = (packets_sent + retransmissions) * 0.0005
        rx_airtime_s = packets_recv * 0.0005
        idle_time_s = max(0.0, total_time - tx_airtime_s - rx_airtime_s)
        
        comm_energy_joules = (
            tx_airtime_s * self.specs.tx_power_w +
            rx_airtime_s * self.specs.rx_power_w +
            idle_time_s * self.specs.idle_power_w
        )

        total_energy_joules = propulsion_energy_joules + avionics_energy_joules + comm_energy_joules
        total_energy_wh = total_energy_joules / 3600.0
        battery_consumed_mah = (total_energy_joules / (self.specs.battery_voltage * 3.6))

        avg_power_watts = total_energy_joules / max(0.01, total_time)

        return {
            "total_energy_joules": round(float(total_energy_joules), 2),
            "propulsion_energy_joules": round(float(propulsion_energy_joules), 2),
            "comm_energy_joules": round(float(comm_energy_joules), 4),
            "avionics_energy_joules": round(float(avionics_energy_joules), 2),
            "total_energy_wh": round(float(total_energy_wh), 3),
            "battery_consumed_mah": round(float(battery_consumed_mah), 2),
            "avg_power_watts": round(float(avg_power_watts), 2)
        }
