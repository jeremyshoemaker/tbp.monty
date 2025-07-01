# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "SensorConfig",
    "RGBDSensorConfig",
    "SemanticSensorConfig",
]

Size = Tuple[int, int]
Vector3 = Tuple[float, float, float]


class SensorConfig(ABC):
    """Abstract base class for MuJoCo sensor configurations.
    
    All MuJoCo sensors should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        sensor_id: str,
        sensor_type: str,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Vector3 = (0.0, 0.0, 0.0),
    ):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.position = position
        self.rotation = rotation

        # Runtime attributes set during initialization
        self._camera_id = None
        self._renderer = None

    @abstractmethod
    def get_spec(self) -> Dict:
        """Return sensor specification for MuJoCo model creation."""
        pass

    @abstractmethod
    def initialize(self, simulator, agent_id: str):
        """Initialize sensor runtime state after model compilation."""
        pass

    @abstractmethod
    def process_observation(self, raw_obs) -> Dict:
        """Process raw MuJoCo sensor data to Monty format."""
        pass

    def get_action_space(self, agent_id: str) -> Dict:
        """Return sensor-specific actions.
        
        Args:
            agent_id: ID of the agent this sensor belongs to
            
        Returns:
            dict: Dictionary of sensor-specific actions
        """
        return {
            f"{agent_id}.set_sensor_pose": {"type": "set_sensor_pose"},
            f"{agent_id}.set_sensor_rotation": {"type": "set_sensor_rotation"},
            f"{agent_id}.set_sensor_pitch": {"type": "set_sensor_pitch"},
        }


class RGBDSensorConfig(SensorConfig):
    """Configuration for RGB-D camera sensor in MuJoCo.
    
    This sensor provides RGB images and depth maps, similar to
    Habitat's RGBD sensor.
    """

    def __init__(
        self,
        sensor_id: str,
        resolution: Size = (64, 64),
        fov: float = 90.0,
        near: float = 0.01,
        far: float = 10.0,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Vector3 = (0.0, 0.0, 0.0),
    ):
        super().__init__(sensor_id, "rgbd", position, rotation)
        self.resolution = resolution
        self.fov = fov
        self.near = near
        self.far = far

    def get_spec(self) -> Dict:
        """Return camera specification for MuJoCo model."""
        return {
            "name": self.sensor_id,
            "type": "fixed",
            "pos": self.position,
            "euler": self.rotation,
            "fovy": self.fov,
            "resolution": self.resolution,
        }

    def initialize(self, simulator, agent_id: str):
        """Initialize camera after MuJoCo model compilation."""
        # Get camera ID from compiled model
        try:
            self._camera_id = simulator.model.camera(self.sensor_id).id
        except KeyError:
            raise RuntimeError(f"Camera '{self.sensor_id}' not found in MuJoCo model")

    def process_observation(self, raw_obs) -> Dict:
        """Process raw MuJoCo camera observations.
        
        Args:
            raw_obs: Raw observation data from MuJoCo renderer
            
        Returns:
            dict: Processed observations with 'rgba' and 'depth' keys
        """
        # Extract RGB and depth from raw observation
        if isinstance(raw_obs, dict):
            rgba = raw_obs.get("rgb", None)
            depth = raw_obs.get("depth", None)
        else:
            # Handle case where raw_obs is directly an image array
            rgba = raw_obs
            depth = None

        result = {}

        if rgba is not None:
            # Ensure RGBA format (add alpha channel if missing)
            if rgba.shape[-1] == 3:
                alpha = np.ones(rgba.shape[:-1] + (1,), dtype=rgba.dtype) * 255
                rgba = np.concatenate([rgba, alpha], axis=-1)
            result["rgba"] = rgba

        if depth is not None:
            result["depth"] = depth

        return result


class SemanticSensorConfig(SensorConfig):
    """Configuration for semantic segmentation sensor in MuJoCo.
    
    This sensor provides semantic segmentation masks, assigning
    object IDs to pixels.
    """

    def __init__(
        self,
        sensor_id: str,
        resolution: Size = (64, 64),
        fov: float = 90.0,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Vector3 = (0.0, 0.0, 0.0),
    ):
        super().__init__(sensor_id, "semantic", position, rotation)
        self.resolution = resolution
        self.fov = fov

    def get_spec(self) -> Dict:
        """Return camera specification for MuJoCo model."""
        return {
            "name": self.sensor_id,
            "type": "fixed",
            "pos": self.position,
            "euler": self.rotation,
            "fovy": self.fov,
            "resolution": self.resolution,
            "mode": "segmentation",
        }

    def initialize(self, simulator, agent_id: str):
        """Initialize camera after MuJoCo model compilation."""
        try:
            self._camera_id = simulator.model.camera(self.sensor_id).id
        except KeyError:
            raise RuntimeError(f"Camera '{self.sensor_id}' not found in MuJoCo model")

    def process_observation(self, raw_obs) -> Dict:
        """Process raw MuJoCo semantic observations.
        
        Args:
            raw_obs: Raw segmentation data from MuJoCo renderer
            
        Returns:
            dict: Processed observations with 'semantic' key
        """
        if isinstance(raw_obs, dict):
            semantic = raw_obs.get("segmentation", raw_obs.get("semantic", None))
        else:
            semantic = raw_obs

        result = {}
        if semantic is not None:
            # Convert to integer semantic IDs
            if semantic.dtype != np.int32:
                semantic = semantic.astype(np.int32)
            result["semantic"] = semantic

        return result


class DepthSensorConfig(SensorConfig):
    """Configuration for depth-only sensor in MuJoCo.
    
    This sensor provides only depth information, useful for
    tactile sensing applications.
    """

    def __init__(
        self,
        sensor_id: str,
        resolution: Size = (64, 64),
        fov: float = 90.0,
        near: float = 0.01,
        far: float = 10.0,
        position: Vector3 = (0.0, 0.0, 0.0),
        rotation: Vector3 = (0.0, 0.0, 0.0),
    ):
        super().__init__(sensor_id, "depth", position, rotation)
        self.resolution = resolution
        self.fov = fov
        self.near = near
        self.far = far

    def get_spec(self) -> Dict:
        """Return camera specification for MuJoCo model."""
        return {
            "name": self.sensor_id,
            "type": "fixed",
            "pos": self.position,
            "euler": self.rotation,
            "fovy": self.fov,
            "resolution": self.resolution,
            "mode": "depth",
        }

    def initialize(self, simulator, agent_id: str):
        """Initialize camera after MuJoCo model compilation."""
        try:
            self._camera_id = simulator.model.camera(self.sensor_id).id
        except KeyError:
            raise RuntimeError(f"Camera '{self.sensor_id}' not found in MuJoCo model")

    def process_observation(self, raw_obs) -> Dict:
        """Process raw MuJoCo depth observations.
        
        Args:
            raw_obs: Raw depth data from MuJoCo renderer
            
        Returns:
            dict: Processed observations with 'depth' key
        """
        if isinstance(raw_obs, dict):
            depth = raw_obs.get("depth", None)
        else:
            depth = raw_obs

        result = {}
        if depth is not None:
            result["depth"] = depth

        return result
