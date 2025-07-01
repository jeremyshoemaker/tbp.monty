# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, MjModel

from tbp.monty.frameworks.actions.actions import (
    Action,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPitch,
    SetAgentPose,
    SetSensorPitch,
    SetSensorPose,
    SetSensorRotation,
    SetYaw,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.environments.embodied_environment import (
    QuaternionWXYZ,
    VectorXYZ,
)

from .actuator import MuJoCoActuator
from .agents import MuJoCoAgent


class MuJoCoSimulator(MuJoCoActuator):
    """Simulator implementation for MuJoCo.

    MuJoCo's data model consists of three parts, a spec defining the scene, a
    model representing a scene generated from a spec, and the associated data or state
    of the simulation based on the model.

    To allow programmatic editing of the scene, we're using an MjSpec that we will
    recompile the model and data from whenever an object is added or removed.

    Attributes:
        agents: List of MuJoCoAgent instances to place in the simulator
        data_path: Path to MuJoCo assets and object libraries
        seed: Random seed for reproducible simulations
        enable_physics: Whether to enable physics simulation
    """

    def __init__(
        self,
        agents: List[MuJoCoAgent],
        data_path: Optional[str] = None,
        seed: int = 42,
        enable_physics: bool = True,
    ) -> None:
        # Initialize MuJoCo spec, model, and data
        self.spec = MjSpec()
        self._setup_basic_scene()

        # Store configuration
        self._agents = agents
        self.data_path = data_path
        self.enable_physics = enable_physics
        self.seed = seed

        # Initialize random number generator
        self.np_rng = np.random.default_rng(seed)

        # Track action space and agent mappings
        self._action_space = set()
        self._agent_id_to_agent = {}

        # Object tracking
        self._object_counter = 0
        self._object_templates = {}

        # Add agents to the scene
        self._setup_agents()

        # Compile initial model and data
        self._recompile()

        # Initialize agents with compiled model
        for agent in self._agents:
            agent.initialize(self)

        # Load object libraries if data path provided
        if data_path is not None:
            self._load_object_libraries()

    def _setup_basic_scene(self):
        """Setup basic MuJoCo scene with default lighting and ground."""
        # Add default lighting
        self.spec.worldbody.add("light", name="main_light", pos=[0, 0, 3])

        # Add ground plane
        self.spec.worldbody.add(
            "geom",
            name="ground",
            type="plane",
            size=[10, 10, 0.1],
            rgba=[0.8, 0.8, 0.8, 1.0]
        )

        # Setup physics options
        if self.enable_physics:
            self.spec.option.gravity = [0, 0, -9.81]
        else:
            self.spec.option.gravity = [0, 0, 0]

    def _setup_agents(self):
        """Add agents to the MuJoCo scene."""
        for agent in self._agents:
            # Update global action space
            agent_actions = agent.get_action_space()
            self._action_space.update(agent_actions.keys())

            # Map agent ID to agent instance
            self._agent_id_to_agent[agent.agent_id] = agent

            # Add agent body to scene
            agent_body = self.spec.worldbody.add(
                "body",
                name=agent.body_name,
                pos=agent.position,
            )

            # Add agent geometry (simple capsule for now)
            agent_body.add(
                "geom",
                name=f"{agent.agent_id}_geom",
                type="capsule",
                size=[0.05, 0.5],
                rgba=[0.2, 0.6, 0.8, 1.0]
            )

            # Add sensors (cameras) to agent body
            for sensor in agent.sensors:
                sensor_spec = sensor.get_spec()
                agent_body.add(
                    "camera",
                    name=sensor_spec["name"],
                    pos=sensor_spec["pos"],
                    euler=sensor_spec["euler"],
                    fovy=sensor_spec["fovy"],
                )

    def _recompile(self):
        """Recompile MuJoCo model and data from current spec."""
        self.model: MjModel = self.spec.compile()
        self.data: MjData = MjData(self.model)

    def _load_object_libraries(self):
        """Load object templates from data path."""
        data_path = Path(self.data_path).expanduser().absolute()

        # Look for common object file formats
        for pattern in ["*.xml", "*.mjcf"]:
            for obj_file in data_path.glob(f"**/{pattern}"):
                try:
                    # Load object spec and store as template
                    obj_name = obj_file.stem
                    self._object_templates[obj_name] = str(obj_file)
                except Exception as e:
                    print(f"Warning: Failed to load object {obj_file}: {e}")

    def initialize_agent(self, agent_id, agent_state):
        """Update agent runtime state.

        Args:
            agent_id: Agent ID to update
            agent_state: New state with position and rotation
        """
        agent = self._agent_id_to_agent.get(agent_id)
        if agent and agent._body_id is not None:
            # Update position
            if hasattr(agent_state, "position"):
                self.data.xpos[agent._body_id] = np.array(agent_state.position)

            # Update rotation
            if hasattr(agent_state, "rotation"):
                if hasattr(agent_state.rotation, "components"):
                    # Quaternion object
                    quat = np.array([
                        agent_state.rotation.w,
                        agent_state.rotation.x,
                        agent_state.rotation.y,
                        agent_state.rotation.z
                    ])
                else:
                    # Array-like quaternion
                    quat = np.array(agent_state.rotation)
                self.data.xquat[agent._body_id] = quat

    def remove_all_objects(self):
        """Remove all objects from the simulated environment."""
        # Clear all added objects by rebuilding scene with only agents
        self.spec = MjSpec()
        self._setup_basic_scene()
        self._setup_agents()
        self._recompile()

        # Re-initialize agents
        for agent in self._agents:
            agent.initialize(self)

        self._object_counter = 0

    def add_object(
        self,
        name: str,
        position: VectorXYZ = (0.0, 0.0, 0.0),
        rotation: QuaternionWXYZ = (1.0, 0.0, 0.0, 0.0),
        scale: VectorXYZ = (1.0, 1.0, 1.0),
        semantic_id: Optional[str] = None,
        enable_physics=False,
        object_to_avoid=False,
        primary_target_bb: Optional[List] = None,
    ) -> None:
        """Add new object to simulated environment.

        Adds a new object based on the named object. This assumes that the set of
        available objects are preloaded and keyed by name.

        Args:
            name (str): Registered object name
            position (VectorXYZ): Initial absolute position of the object
            rotation (QuaternionWXYZ): Initial orientation of the object
            scale (VectorXYZ): Initial object scale
            semantic_id (Optional[str]): Optional override object semantic ID
            enable_physics (bool): Whether to enable physics on the object
            object_to_avoid (bool): If True, ensure the object is not colliding with
              other objects
            primary_target_bb (List | None): If not None, this is a list of the min and
              max corners of a bounding box for the primary object, used to prevent
              obscuring the primary objet with the new object.
        """
        # Create unique object name
        obj_name = f"{name}_{self._object_counter}"

        # Add object to scene spec
        if name in self._object_templates:
            # Load from template file
            template_path = self._object_templates[name]
            # TODO: Implement loading from external MJCF files
            # For now, create basic geometric object
            self._add_primitive_object(obj_name, name, position, rotation, scale, semantic_id)
        else:
            # Create primitive geometric object
            self._add_primitive_object(obj_name, name, position, rotation, scale, semantic_id)

        # Recompile model to include new object
        self._recompile()

        # Re-initialize agents after recompilation
        for agent in self._agents:
            agent.initialize(self)

        # Handle collision avoidance if requested
        if object_to_avoid and self.enable_physics:
            self._handle_object_collision_avoidance(obj_name, position, primary_target_bb)

        self._object_counter += 1

    def _handle_object_collision_avoidance(self, obj_name, start_position, primary_target_bb):
        """Handle collision avoidance for newly added object."""
        # Find the object body in the compiled model
        try:
            obj_body_id = self.model.body(obj_name).id
        except KeyError:
            return  # Object not found, skip collision handling

        max_attempts = 100
        step_size = 0.1

        for attempt in range(max_attempts):
            # Step physics to check for collisions
            mujoco.mj_step(self.model, self.data)

            # Check for contacts involving this object
            has_collision = False
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]

                if geom1_body == obj_body_id or geom2_body == obj_body_id:
                    has_collision = True
                    break

            if not has_collision:
                # No collision, position is good
                break

            # Move object to new position
            direction = self.np_rng.uniform(-1, 1, size=3)
            direction /= np.linalg.norm(direction)

            new_pos = np.array(start_position) + direction * step_size * attempt
            self.data.xpos[obj_body_id] = new_pos

    def _add_primitive_object(self, obj_name, shape_type, position, rotation, scale, semantic_id):
        """Add a primitive geometric object to the scene."""
        # Create object body
        obj_body = self.spec.worldbody.add(
            "body",
            name=obj_name,
            pos=position,
        )

        # Convert quaternion to euler for MuJoCo
        if isinstance(rotation, (list, tuple)) and len(rotation) == 4:
            # Convert quaternion (w,x,y,z) to euler angles
            w, x, y, z = rotation
            # Simple conversion for now - more robust conversion would use mujoco functions
            euler = [0, 0, 2 * np.arctan2(y, w)]  # Approximate yaw only
        else:
            euler = [0, 0, 0]

        obj_body.euler = euler

        # Add geometry based on shape type
        if shape_type.lower() in ["cube", "box", "cubesolid"]:
            obj_body.add(
                "geom",
                name=f"{obj_name}_geom",
                type="box",
                size=np.array(scale) * 0.5,  # MuJoCo box size is half-extents
                rgba=[0.8, 0.2, 0.2, 1.0]
            )
        elif shape_type.lower() in ["sphere", "uvspheresold", "icospheresold"]:
            obj_body.add(
                "geom",
                name=f"{obj_name}_geom",
                type="sphere",
                size=[scale[0] * 0.5],  # Use first scale component as radius
                rgba=[0.2, 0.8, 0.2, 1.0]
            )
        elif shape_type.lower() in ["cylinder", "cylindersolid"]:
            obj_body.add(
                "geom",
                name=f"{obj_name}_geom",
                type="cylinder",
                size=[scale[0] * 0.5, scale[1] * 0.5],  # radius, half-height
                rgba=[0.2, 0.2, 0.8, 1.0]
            )
        elif shape_type.lower() in ["cone", "conesolid"]:
            # MuJoCo doesn't have cone primitive, use cylinder with tapered top
            obj_body.add(
                "geom",
                name=f"{obj_name}_geom",
                type="cylinder",
                size=[scale[0] * 0.5, scale[1] * 0.5],
                rgba=[0.8, 0.8, 0.2, 1.0]
            )
        else:
            # Default to box
            obj_body.add(
                "geom",
                name=f"{obj_name}_geom",
                type="box",
                size=np.array(scale) * 0.5,
                rgba=[0.5, 0.5, 0.5, 1.0]
            )

    def get_num_objects(self) -> int:
        """Return the number of instantiated objects in the environment."""
        return self._object_counter

    def get_action_space(self):
        """Returns the set of all available actions."""
        return set(self._action_space)

    def get_agent(self, agent_id):
        """Return agent instance."""
        return self._agent_id_to_agent.get(agent_id)

    def get_observations(self):
        """Get sensor observations."""
        # Step physics simulation first
        mujoco.mj_step(self.model, self.data)

        # Collect observations from all agent sensors
        observations = {}

        for agent in self._agents:
            agent_obs = {}

            for sensor in agent.sensors:
                if sensor._camera_id is not None:
                    # Render camera observations
                    sensor_obs = self._render_camera_observation(sensor)
                    processed_obs = sensor.process_observation(sensor_obs)
                    agent_obs[sensor.sensor_id] = processed_obs

            observations[agent.agent_id] = agent_obs

        return observations

    def _render_camera_observation(self, sensor):
        """Render observation from a camera sensor."""
        # Create renderer if needed
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model)

        # Update renderer camera
        self._renderer.update_scene(self.data, camera=sensor._camera_id)

        # Render RGB image
        rgb_img = self._renderer.render()

        # Render depth if needed
        depth_img = None
        if sensor.sensor_type in ["rgbd", "depth"]:
            # Enable depth rendering
            self._renderer.enable_depth_rendering()
            depth_img = self._renderer.render()

        # Return observation dictionary
        result = {"rgb": rgb_img}
        if depth_img is not None:
            result["depth"] = depth_img

        return result

    def get_states(self):
        """Get agent and sensor states."""
        states = {}

        for agent in self._agents:
            if agent._body_id is not None:
                # Get agent body state
                pos = self.data.xpos[agent._body_id].copy()
                quat = self.data.xquat[agent._body_id].copy()

                # Get sensor states
                sensor_states = {}
                for sensor in agent.sensors:
                    if sensor._camera_id is not None:
                        # Camera position and orientation relative to agent
                        cam_pos = self.model.cam_pos[sensor._camera_id].copy()
                        cam_quat = getattr(self.model, "cam_quat", [1, 0, 0, 0])

                        sensor_states[sensor.sensor_id] = {
                            "position": cam_pos,
                            "rotation": cam_quat,
                        }

                states[agent.agent_id] = {
                    "position": pos,
                    "rotation": quat,
                    "sensors": sensor_states,
                }

        return states

    def apply_action(self, action: Action) -> Dict[str, Dict]:
        """Execute the given action in the environment.

        Args:
            action (Action): the action to execute

        Returns:
            (Dict[str, Dict]): A dictionary with the observations grouped by agent_id
        """
        action_name = self.action_name(action)
        if action_name not in self._action_space:
            raise ValueError(f"Invalid action name: {action_name}")

        # Execute action based on type
        if isinstance(action, MoveForward):
            self.actuate_move_forward(action)
        elif isinstance(action, TurnLeft):
            self.actuate_turn_left(action)
        elif isinstance(action, TurnRight):
            self.actuate_turn_right(action)
        elif isinstance(action, LookUp):
            self.actuate_look_up(action)
        elif isinstance(action, LookDown):
            self.actuate_look_down(action)
        elif isinstance(action, SetAgentPose):
            self.actuate_set_agent_pose(action)
        elif isinstance(action, SetSensorPose):
            self.actuate_set_sensor_pose(action)
        elif isinstance(action, SetSensorRotation):
            self.actuate_set_sensor_rotation(action)
        elif isinstance(action, MoveTangentially):
            self.actuate_move_tangentially(action)
        elif isinstance(action, OrientHorizontal):
            self.actuate_orient_horizontal(action)
        elif isinstance(action, OrientVertical):
            self.actuate_orient_vertical(action)
        elif isinstance(action, SetYaw):
            self.actuate_set_yaw(action)
        elif isinstance(action, SetAgentPitch):
            self.actuate_set_agent_pitch(action)
        elif isinstance(action, SetSensorPitch):
            self.actuate_set_sensor_pitch(action)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        # Return updated observations
        return self.get_observations()

    def reset(self):
        """Reset the simulator."""
        # Reset MuJoCo data to initial state
        mujoco.mj_resetData(self.model, self.data)

        # Get initial observations
        return self.get_observations()

    def close(self):
        """Close any resources used by the simulator."""
        # Clean up renderer if it exists
        if hasattr(self, "_renderer"):
            del self._renderer

        # Clean up model and data
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "data"):
            del self.data

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
