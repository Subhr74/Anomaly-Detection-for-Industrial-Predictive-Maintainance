"""
PredictX AI - Industrial Telemetry Simulator
Generates multivariate time-series sensor data incorporating Gaussian noise,
continuous thermal & mechanical physics, degradation trends, and fault injection mechanisms.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, List

FEATURE_NAMES = [
    "vibration_rms",
    "bearing_temp",
    "acoustic_emission",
    "hydraulic_pressure",
    "spindle_rpm",
    "load_torque"
]


class IndustrialTelemetrySimulator:
    """
    Simulates high-fidelity telemetry from an industrial CNC milling spindle / pump.
    Maintains continuous physics state with inertia, stochastic noise, and wear degradation.
    Supports on-the-fly fault injection and custom slider overrides.
    """
    def __init__(self, machine_id: str = "CNC-SPINDLE-VF4-04"):
        self.machine_id = machine_id
        self.step_count = 0
        self.operating_hours = 420.0
        self.true_rul_hours = 780.0
        
        # Physics state variables
        self.base_vibration = 1.65
        self.base_temp = 51.5
        self.base_acoustic = 42.0
        self.base_pressure = 150.0
        self.base_rpm = 3000.0
        self.base_torque = 48.0

        # Current smooth state
        self.current_temp = self.base_temp
        self.current_vibration = self.base_vibration
        self.current_pressure = self.base_pressure
        self.current_torque = self.base_torque
        self.current_acoustic = self.base_acoustic
        self.current_rpm = self.base_rpm

        # Injected fault mode: 0: Normal, 1: Bearing, 2: Misalignment, 3: Cavitation, 4: Overheating
        self.active_fault_mode: int = 0
        self.fault_intensity: float = 1.0  # 0.0 to 1.0
        
        # Manual parameter overrides from stress-test sandbox
        self.manual_overrides: Dict[str, Optional[float]] = {name: None for name in FEATURE_NAMES}

    def set_fault_mode(self, mode: int, intensity: float = 1.0):
        """Injects a specific fault mode."""
        self.active_fault_mode = mode
        self.fault_intensity = max(0.1, min(1.0, intensity))

    def clear_fault(self):
        """Resets machine to healthy baseline."""
        self.active_fault_mode = 0
        self.fault_intensity = 1.0
        self.manual_overrides = {name: None for name in FEATURE_NAMES}

    def set_manual_override(self, feature: str, value: Optional[float]):
        """Sets a manual override value for an individual sensor."""
        if feature in self.manual_overrides:
            self.manual_overrides[feature] = value

    def clear_manual_overrides(self):
        """Clears all manual overrides."""
        self.manual_overrides = {name: None for name in FEATURE_NAMES}

    def step(self) -> Dict[str, Any]:
        """
        Advances the simulator by 1 time step and returns the multivariate sensor snapshot.
        Incorporates periodic machining cycles, stochastic noise, and injected faults.
        """
        self.step_count += 1
        self.operating_hours += 0.005  # Slight operational progression
        
        # Periodic machining cycle effect (cutting load oscillation)
        t = self.step_count * 0.15
        cutting_cycle = 0.5 * np.sin(t * 0.4) + 0.3 * np.cos(t * 0.9)

        # Baseline noise
        vib_noise = np.random.normal(0, 0.08)
        temp_noise = np.random.normal(0, 0.12)
        acoust_noise = np.random.normal(0, 0.8)
        press_noise = np.random.normal(0, 0.4)
        rpm_noise = np.random.normal(0, 4.0)
        torque_noise = np.random.normal(0, 0.7)

        # Natural wear degradation trend (as hours increase, baseline rises slightly)
        wear_factor = min(0.35, (self.operating_hours / 2500.0))
        
        # Fault injection contributions
        fault_vib = 0.0
        fault_temp = 0.0
        fault_acoust = 0.0
        fault_press = 0.0
        fault_rpm = 0.0
        fault_torque = 0.0

        if self.active_fault_mode == 1:
            # 1: Bearing Inner Race Fault (high-frequency ultrasonic spikes + severe vibration)
            fault_vib = 4.2 * self.fault_intensity + np.random.exponential(0.6)
            fault_acoust = 32.0 * self.fault_intensity + np.random.normal(0, 4.0)
            fault_temp = 9.5 * self.fault_intensity
            fault_torque = 5.0 * self.fault_intensity
        elif self.active_fault_mode == 2:
            # 2: Shaft Misalignment (2X RPM vibration peak, thermal friction, torque surging)
            fault_vib = 3.6 * self.fault_intensity + 0.8 * np.sin(t * 3.0)
            fault_temp = 16.0 * self.fault_intensity
            fault_torque = 24.0 * self.fault_intensity
            fault_rpm = -55.0 * self.fault_intensity
            fault_acoust = 12.0 * self.fault_intensity
        elif self.active_fault_mode == 3:
            # 3: Hydraulic Cavitation (pressure drop, extreme pressure ripple, acoustic crackle)
            fault_press = -38.0 * self.fault_intensity + np.random.normal(0, 7.5)
            fault_acoust = 26.0 * self.fault_intensity
            fault_vib = 1.9 * self.fault_intensity
            fault_torque = 11.0 * self.fault_intensity * np.sin(t * 2.5)
        elif self.active_fault_mode == 4:
            # 4: Spindle Overheating / Thermal Runaway (drastic temp climb, thermal expansion binding)
            fault_temp = 36.0 * self.fault_intensity
            fault_torque = 28.0 * self.fault_intensity
            fault_vib = 2.1 * self.fault_intensity
            fault_acoust = 16.0 * self.fault_intensity
            fault_rpm = -40.0 * self.fault_intensity

        # Target values
        target_vibration = self.base_vibration + wear_factor * 1.5 + cutting_cycle * 0.2 + vib_noise + fault_vib
        target_temp = self.base_temp + wear_factor * 8.0 + cutting_cycle * 1.2 + temp_noise + fault_temp
        target_acoustic = self.base_acoustic + wear_factor * 6.0 + cutting_cycle * 1.5 + acoust_noise + fault_acoust
        target_pressure = self.base_pressure - wear_factor * 5.0 + press_noise + fault_press
        target_rpm = self.base_rpm - wear_factor * 20.0 + rpm_noise + fault_rpm
        target_torque = self.base_torque + wear_factor * 8.0 + cutting_cycle * 3.0 + torque_noise + fault_torque

        # Thermal inertia: temperature changes gradually via low-pass filter
        self.current_temp = 0.92 * self.current_temp + 0.08 * target_temp
        self.current_vibration = 0.65 * self.current_vibration + 0.35 * target_vibration
        self.current_pressure = 0.70 * self.current_pressure + 0.30 * target_pressure
        self.current_acoustic = 0.50 * self.current_acoustic + 0.50 * target_acoustic
        self.current_torque = 0.65 * self.current_torque + 0.35 * target_torque
        self.current_rpm = 0.60 * self.current_rpm + 0.40 * target_rpm

        # Check manual overrides
        final_vibration = self.manual_overrides["vibration_rms"] if self.manual_overrides["vibration_rms"] is not None else self.current_vibration
        final_temp = self.manual_overrides["bearing_temp"] if self.manual_overrides["bearing_temp"] is not None else self.current_temp
        final_acoustic = self.manual_overrides["acoustic_emission"] if self.manual_overrides["acoustic_emission"] is not None else self.current_acoustic
        final_pressure = self.manual_overrides["hydraulic_pressure"] if self.manual_overrides["hydraulic_pressure"] is not None else self.current_pressure
        final_rpm = self.manual_overrides["spindle_rpm"] if self.manual_overrides["spindle_rpm"] is not None else self.current_rpm
        final_torque = self.manual_overrides["load_torque"] if self.manual_overrides["load_torque"] is not None else self.current_torque

        # Physical safety clamps
        final_vibration = max(0.4, float(final_vibration))
        final_temp = max(20.0, float(final_temp))
        final_acoustic = max(10.0, float(final_acoustic))
        final_pressure = max(60.0, float(final_pressure))
        final_rpm = max(1200.0, float(final_rpm))
        final_torque = max(10.0, float(final_torque))

        # Calculate instantaneous RUL degradation
        stress_multiplier = 1.0
        if self.active_fault_mode > 0:
            stress_multiplier = 4.5
        calculated_rul = max(8.0, 1000.0 - (self.operating_hours * 0.95 * stress_multiplier))
        if self.active_fault_mode == 1:
            calculated_rul = min(calculated_rul, 38.0)
        elif self.active_fault_mode == 4:
            calculated_rul = min(calculated_rul, 14.0)

        snapshot = {
            "timestamp": time.time(),
            "machine_id": self.machine_id,
            "step_count": self.step_count,
            "operating_hours": round(self.operating_hours, 1),
            "simulated_rul_hours": round(calculated_rul, 1),
            "injected_fault_mode": self.active_fault_mode,
            "telemetry": {
                "vibration_rms": round(final_vibration, 3),
                "bearing_temp": round(final_temp, 2),
                "acoustic_emission": round(final_acoustic, 2),
                "hydraulic_pressure": round(final_pressure, 2),
                "spindle_rpm": round(final_rpm, 1),
                "load_torque": round(final_torque, 2)
            },
            "feature_vector": [
                round(final_vibration, 4),
                round(final_temp, 4),
                round(final_acoustic, 4),
                round(final_pressure, 4),
                round(final_rpm, 4),
                round(final_torque, 4)
            ]
        }
        return snapshot

    @staticmethod
    def generate_training_dataset(n_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes a rich, balanced training dataset spanning all 5 operational classes:
        0: Normal (700 samples)
        1: Bearing Inner Race Fault (200 samples)
        2: Shaft Misalignment (200 samples)
        3: Hydraulic Cavitation (200 samples)
        4: Spindle Overheating (200 samples)

        Returns:
            X (features, shape [N, 6]),
            y_class (fault classes, shape [N]),
            y_rul (RUL in hours, shape [N])
        """
        np.random.seed(42)
        X_list = []
        y_class_list = []
        y_rul_list = []

        # 0. Normal Operation (varying loads: idle, medium, heavy roughing)
        n_normal = int(n_samples * 0.5)
        for _ in range(n_normal):
            vib = np.random.normal(1.65, 0.35)
            temp = np.random.normal(51.0, 4.0)
            acoust = np.random.normal(42.0, 3.8)
            press = np.random.normal(150.0, 3.5)
            rpm = np.random.normal(3000.0, 25.0)
            torque = np.random.normal(48.0, 4.5)
            
            rul = np.random.uniform(500.0, 1100.0)
            X_list.append([vib, temp, acoust, press, rpm, torque])
            y_class_list.append(0)
            y_rul_list.append(rul)

        # 1. Bearing Inner Race Fault (High acoustic spikes, high vibration)
        n_bearing = int(n_samples * 0.125)
        for _ in range(n_bearing):
            vib = np.random.normal(5.8, 0.85)
            temp = np.random.normal(64.0, 4.5)
            acoust = np.random.normal(74.0, 5.0)
            press = np.random.normal(148.0, 4.0)
            rpm = np.random.normal(2980.0, 35.0)
            torque = np.random.normal(54.0, 5.0)
            
            rul = np.random.uniform(15.0, 120.0)
            X_list.append([vib, temp, acoust, press, rpm, torque])
            y_class_list.append(1)
            y_rul_list.append(rul)

        # 2. Shaft Misalignment (Elevated vibration, high torque drag, warm bearings)
        n_misalign = int(n_samples * 0.125)
        for _ in range(n_misalign):
            vib = np.random.normal(4.8, 0.65)
            temp = np.random.normal(68.0, 4.0)
            acoust = np.random.normal(56.0, 4.5)
            press = np.random.normal(147.0, 4.0)
            rpm = np.random.normal(2930.0, 45.0)
            torque = np.random.normal(74.0, 6.0)
            
            rul = np.random.uniform(40.0, 180.0)
            X_list.append([vib, temp, acoust, press, rpm, torque])
            y_class_list.append(2)
            y_rul_list.append(rul)

        # 3. Hydraulic Cavitation (Pressure collapse, high variance, acoustic crackle)
        n_cavit = int(n_samples * 0.125)
        for _ in range(n_cavit):
            vib = np.random.normal(3.2, 0.5)
            temp = np.random.normal(56.0, 3.5)
            acoust = np.random.normal(67.0, 4.8)
            press = np.random.normal(112.0, 7.0)  # Distinct pressure drop
            rpm = np.random.normal(2985.0, 30.0)
            torque = np.random.normal(58.0, 6.5)
            
            rul = np.random.uniform(50.0, 220.0)
            X_list.append([vib, temp, acoust, press, rpm, torque])
            y_class_list.append(3)
            y_rul_list.append(rul)

        # 4. Spindle Overheating / Thermal Runaway (Extreme temperature climb, high torque)
        n_overheat = int(n_samples * 0.125)
        for _ in range(n_overheat):
            vib = np.random.normal(3.8, 0.6)
            temp = np.random.normal(88.0, 5.0)  # Extreme temperature
            acoust = np.random.normal(58.0, 4.0)
            press = np.random.normal(142.0, 5.0)
            rpm = np.random.normal(2940.0, 40.0)
            torque = np.random.normal(78.0, 7.0)
            
            rul = np.random.uniform(4.0, 45.0)
            X_list.append([vib, temp, acoust, press, rpm, torque])
            y_class_list.append(4)
            y_rul_list.append(rul)

        X = np.array(X_list, dtype=float)
        y_class = np.array(y_class_list, dtype=int)
        y_rul = np.array(y_rul_list, dtype=float)

        # Shuffle
        shuffle_idx = np.random.permutation(len(y_class))
        return X[shuffle_idx], y_class[shuffle_idx], y_rul[shuffle_idx]

