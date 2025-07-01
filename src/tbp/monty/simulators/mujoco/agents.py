# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import uuid
from typing import Dict, List, Tuple

from .sensors import SensorConfig

__all__ = [
    "MuJoCoAgent",
    "SingleSensorAgent",
    "MultiSensorAgent",
]

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Size = Tuple[int, int]


class MuJoCoAgent:
    """MuJoCo agent wrapper.

    Agents are used to define moveable bodies in the MuJoCo environment.
    Every MuJoCo agent will inherit from this class.

    Attributes:
        agent_id: Unique ID of this agent in env. Observations returned by environment
            will be mapped to this id. Actions provided by this agent will be
            prefixed by this id.
        position: Agent initial position in meters. Default (0, 1.5, 0)
        rotation: Agent initial rotation quaternion. Default (1, 0, 0, 0)
        body_name: Name of the MuJoCo body this agent controls
    """

    def __init__(
        self,
        agent_id: str,
        position: Vector3 = (0.0, 1.5, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        body_name: str = "agent_body",
    ):
        if agent_id is None:
            agent_id = uuid.uuid4().hex
        self.agent_id = agent_id
        self.position = position
        self.rotation = rotation
        self.body_name = body_name
        self.sensors: List[SensorConfig] = []

        # MuJoCo-specific attributes
        self._body_id = None
        self._joint_ids = {}

    def get_spec(self):
        """Returns a MuJoCo agent specification.

        Returns:
            dict: Configuration dictionary for creating this agent in MuJoCo
        """
        return {
            "agent_id": self.agent_id,
            "body_name": self.body_name,
            "position": self.position,
            "rotation": self.rotation,
            "sensors": [sensor.get_spec() for sensor in self.sensors],
        }

    def initialize(self, simulator):
        """Initialize MuJoCo agent runtime state.

        This method must be called to update the agent and sensors runtime
        instance after the MuJoCo model has been compiled.

        Args:
            simulator: Instantiated MuJoCoSimulator instance
        """
        # Get body ID from compiled model
        self._body_id = simulator.model.body(self.body_name).id

        # Initialize sensor runtime state
        for sensor in self.sensors:
            sensor.initialize(simulator, self.agent_id)

    def process_observations(self, agent_obs):
        """Process raw MuJoCo observations to Monty-compatible format.

        Args:
            agent_obs: Agent raw MuJoCo observations

        Returns:
            dict: Processed observations grouped by sensor_id
        """
        processed_obs = {}

        for sensor in self.sensors:
            sensor_id = sensor.sensor_id
            if sensor_id in agent_obs:
                processed_obs[sensor_id] = sensor.process_observation(
                    agent_obs[sensor_id]
                )

        return processed_obs

    def get_action_space(self):
        """Returns the action space for this agent.

        Returns:
            dict: Dictionary mapping action names to action specifications
        """
        # Base action space for agent movement
        action_space = {
            f"{self.agent_id}.move_forward": {"type": "move_forward"},
            f"{self.agent_id}.turn_left": {"type": "turn_left"},
            f"{self.agent_id}.turn_right": {"type": "turn_right"},
            f"{self.agent_id}.look_up": {"type": "look_up"},
            f"{self.agent_id}.look_down": {"type": "look_down"},
            f"{self.agent_id}.set_agent_pose": {"type": "set_agent_pose"},
        }

        # Add sensor-specific actions
        for sensor in self.sensors:
            sensor_actions = sensor.get_action_space(self.agent_id)
            action_space.update(sensor_actions)

        return action_space


class SingleSensorAgent(MuJoCoAgent):
    """MuJoCo agent with a single sensor.

    This is a convenience class for creating agents with a single sensor,
    similar to the common use case in Habitat.
    """

    def __init__(
        self,
        agent_id: str,
        sensor_id: str,
        sensor_type: str = "rgbd",
        resolution: Size = (64, 64),
        position: Vector3 = (0.0, 1.5, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
        **sensor_kwargs,
    ):
        super().__init__(agent_id, position, rotation)

        # Import sensor config class dynamically based on type
        if sensor_type == "rgbd":
            from .sensors import RGBDSensorConfig
            sensor_config = RGBDSensorConfig(
                sensor_id=sensor_id,
                resolution=resolution,
                **sensor_kwargs,
            )
        elif sensor_type == "semantic":
            from .sensors import SemanticSensorConfig
            sensor_config = SemanticSensorConfig(
                sensor_id=sensor_id,
                resolution=resolution,
                **sensor_kwargs,
            )
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        self.sensors.append(sensor_config)


class MultiSensorAgent(MuJoCoAgent):
    """MuJoCo agent with multiple sensors.

    This class allows creating agents with multiple sensors of different types,
    useful for multi-modal sensing scenarios.
    """

    def __init__(
        self,
        agent_id: str,
        sensor_configs: List[Dict],
        position: Vector3 = (0.0, 1.5, 0.0),
        rotation: Quaternion = (1.0, 0.0, 0.0, 0.0),
    ):
        super().__init__(agent_id, position, rotation)

        # Create sensor configurations
        for config in sensor_configs:
            sensor_type = config.get("type", "rgbd")
            sensor_id = config["sensor_id"]
            resolution = config.get("resolution", (64, 64))

            if sensor_type == "rgbd":
                from .sensors import RGBDSensorConfig
                sensor_config = RGBDSensorConfig(
                    sensor_id=sensor_id,
                    resolution=resolution,
                    **{k: v for k, v in config.items()
                       if k not in ["type", "sensor_id", "resolution"]},
                )
            elif sensor_type == "semantic":
                from .sensors import SemanticSensorConfig
                sensor_config = SemanticSensorConfig(
                    sensor_id=sensor_id,
                    resolution=resolution,
                    **{k: v for k, v in config.items()
                       if k not in ["type", "sensor_id", "resolution"]},
                )
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")

            self.sensors.append(sensor_config)
