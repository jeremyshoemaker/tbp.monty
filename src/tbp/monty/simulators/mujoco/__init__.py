# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""MuJoCo simulator implementation for Monty.

This module provides a MuJoCo-based simulator that can be used as a drop-in
replacement for HabitatSim in Monty experiments.

Example usage:
    camera = SingleSensorAgent(
        agent_id="camera",
        sensor_id="camera_id", 
        resolution=(64, 64),
    )
    
    with MuJoCoSimulator(agents=[camera]) as sim:
        sim.add_object(name="cube", position=(0.0, 1.5, -0.2))
        obs = sim.get_observations()
"""

from .actuator import MuJoCoActuator
from .agents import MuJoCoAgent, MultiSensorAgent, SingleSensorAgent
from .sensors import (
    DepthSensorConfig,
    RGBDSensorConfig,
    SemanticSensorConfig,
    SensorConfig,
)
from .simulator import MuJoCoSimulator

__all__ = [
    "MuJoCoSimulator",
    "MuJoCoAgent",
    "SingleSensorAgent",
    "MultiSensorAgent",
    "SensorConfig",
    "RGBDSensorConfig",
    "SemanticSensorConfig",
    "DepthSensorConfig",
    "MuJoCoActuator",
]
