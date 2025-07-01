# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import numpy as np
from typing_extensions import Protocol

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

__all__ = [
    "MuJoCoActuator",
    "MuJoCoActuatorRequirements",
]


class MuJoCoActuatorRequirements(Protocol):
    """MuJoCoActuator requires these to be available when mixed in."""

    def get_agent(self, agent_id: str): ...
    model: object  # MjModel
    data: object   # MjData


class MuJoCoActuator(MuJoCoActuatorRequirements):
    """MuJoCo implementation of an Actuator.

    MuJoCoActuator is responsible for executing actions in the MuJoCo simulation.
    It translates Monty actions into MuJoCo operations on bodies, joints, and cameras.

    This class is expected to be mixed into MuJoCoSimulator and expects
    MuJoCoActuatorRequirements to be met.
    """

    def action_name(self, action: Action) -> str:
        """Returns Monty's MuJoCo action naming convention.

        The action name is prefixed by the agent ID.
        """
        return f"{action.agent_id}.{action.name}"

    def actuate_move_forward(self, action: MoveForward) -> None:
        """Move agent forward by specified distance."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Get current position and orientation
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()

            # Convert quaternion to rotation matrix to get forward direction
            rot_mat = np.zeros((3, 3))
            from mujoco import mju_quat2Mat
            mju_quat2Mat(rot_mat, quat)

            # Forward direction (assuming -Z is forward in MuJoCo)
            forward = -rot_mat[:, 2]

            # Update position
            new_pos = pos + forward * action.distance

            # Set new position (this assumes the body is freely movable)
            self.data.xpos[body_id] = new_pos

    def actuate_turn_left(self, action: TurnLeft) -> None:
        """Turn agent left by specified angle."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Get current quaternion
            quat = self.data.xquat[body_id].copy()

            # Create rotation quaternion for left turn (positive Y rotation)
            angle_rad = np.radians(action.rotation_degrees)
            sin_half = np.sin(angle_rad / 2)
            cos_half = np.cos(angle_rad / 2)
            turn_quat = np.array([cos_half, 0, sin_half, 0])

            # Multiply quaternions to combine rotations
            from mujoco import mju_mulQuat
            new_quat = np.zeros(4)
            mju_mulQuat(new_quat, quat, turn_quat)

            # Set new orientation
            self.data.xquat[body_id] = new_quat

    def actuate_turn_right(self, action: TurnRight) -> None:
        """Turn agent right by specified angle."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Get current quaternion
            quat = self.data.xquat[body_id].copy()

            # Create rotation quaternion for right turn (negative Y rotation)
            angle_rad = np.radians(-action.rotation_degrees)
            sin_half = np.sin(angle_rad / 2)
            cos_half = np.cos(angle_rad / 2)
            turn_quat = np.array([cos_half, 0, sin_half, 0])

            # Multiply quaternions to combine rotations
            from mujoco import mju_mulQuat
            new_quat = np.zeros(4)
            mju_mulQuat(new_quat, quat, turn_quat)

            # Set new orientation
            self.data.xquat[body_id] = new_quat

    def actuate_look_up(self, action: LookUp) -> None:
        """Tilt agent/sensor up by specified angle."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Get current quaternion
            quat = self.data.xquat[body_id].copy()

            # Create rotation quaternion for upward tilt (negative X rotation)
            angle_rad = np.radians(-action.rotation_degrees)
            sin_half = np.sin(angle_rad / 2)
            cos_half = np.cos(angle_rad / 2)
            tilt_quat = np.array([cos_half, sin_half, 0, 0])

            # Apply constraint if specified
            if hasattr(action, "constraint_degrees") and action.constraint_degrees:
                # TODO: Implement constraint checking
                pass

            # Multiply quaternions to combine rotations
            from mujoco import mju_mulQuat
            new_quat = np.zeros(4)
            mju_mulQuat(new_quat, quat, tilt_quat)

            # Set new orientation
            self.data.xquat[body_id] = new_quat

    def actuate_look_down(self, action: LookDown) -> None:
        """Tilt agent/sensor down by specified angle."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Get current quaternion
            quat = self.data.xquat[body_id].copy()

            # Create rotation quaternion for downward tilt (positive X rotation)
            angle_rad = np.radians(action.rotation_degrees)
            sin_half = np.sin(angle_rad / 2)
            cos_half = np.cos(angle_rad / 2)
            tilt_quat = np.array([cos_half, sin_half, 0, 0])

            # Apply constraint if specified
            if hasattr(action, "constraint_degrees") and action.constraint_degrees:
                # TODO: Implement constraint checking
                pass

            # Multiply quaternions to combine rotations
            from mujoco import mju_mulQuat
            new_quat = np.zeros(4)
            mju_mulQuat(new_quat, quat, tilt_quat)

            # Set new orientation
            self.data.xquat[body_id] = new_quat

    def actuate_set_agent_pose(self, action: SetAgentPose) -> None:
        """Set agent to specific pose (position and orientation)."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Set position
            if hasattr(action, "position") and action.position is not None:
                self.data.xpos[body_id] = np.array(action.position)

            # Set orientation
            if hasattr(action, "rotation") and action.rotation is not None:
                if isinstance(action.rotation, (list, tuple)) and len(action.rotation) == 4:
                    # Quaternion format (w, x, y, z)
                    self.data.xquat[body_id] = np.array(action.rotation)
                else:
                    # Euler angles - convert to quaternion
                    from mujuco import mju_euler2Quat
                    quat = np.zeros(4)
                    mju_euler2Quat(quat, np.array(action.rotation))
                    self.data.xquat[body_id] = quat

    def actuate_set_sensor_pose(self, action: SetSensorPose) -> None:
        """Set sensor to specific pose relative to agent."""
        agent = self.get_agent(action.agent_id)

        # Find the sensor camera
        sensor_id = getattr(action, "sensor_id", None)
        if sensor_id:
            for sensor in agent.sensors:
                if sensor.sensor_id == sensor_id and sensor._camera_id is not None:
                    # Update camera position and orientation in the model
                    # Note: This may require model recompilation depending on implementation
                    camera_id = sensor._camera_id

                    if hasattr(action, "position") and action.position is not None:
                        # Set camera position relative to agent body
                        self.model.cam_pos[camera_id] = np.array(action.position)

                    if hasattr(action, "rotation") and action.rotation is not None:
                        # Set camera orientation
                        if isinstance(action.rotation, (list, tuple)) and len(action.rotation) == 3:
                            # Euler angles
                            self.model.cam_quat[camera_id] = np.array(action.rotation)
                    break

    def actuate_set_sensor_rotation(self, action: SetSensorRotation) -> None:
        """Set sensor rotation."""
        agent = self.get_agent(action.agent_id)

        sensor_id = getattr(action, "sensor_id", None)
        if sensor_id:
            for sensor in agent.sensors:
                if sensor.sensor_id == sensor_id and sensor._camera_id is not None:
                    camera_id = sensor._camera_id

                    if hasattr(action, "rotation") and action.rotation is not None:
                        # Set camera rotation
                        self.model.cam_quat[camera_id] = np.array(action.rotation)
                    break

    def actuate_move_tangentially(self, action: MoveTangentially) -> None:
        """Move agent tangentially to current surface or trajectory."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Get current position and orientation
            pos = self.data.xpos[body_id].copy()
            quat = self.data.xquat[body_id].copy()

            # Convert quaternion to rotation matrix
            rot_mat = np.zeros((3, 3))
            from mujoco import mju_quat2Mat
            mju_quat2Mat(rot_mat, quat)

            # Tangential direction (assuming X is right/tangential)
            tangent = rot_mat[:, 0]

            # Apply direction multiplier if specified
            direction = getattr(action, "direction", 1.0)
            distance = getattr(action, "distance", 0.1)

            # Update position
            new_pos = pos + tangent * direction * distance
            self.data.xpos[body_id] = new_pos

    def actuate_orient_horizontal(self, action: OrientHorizontal) -> None:
        """Orient agent horizontally."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Get current quaternion and extract yaw component
            quat = self.data.xquat[body_id].copy()

            # Convert to Euler, zero out pitch and roll, keep yaw
            from mujoco import mju_quat2Euler
            euler = np.zeros(3)
            mju_quat2Euler(euler, quat)

            # Keep only yaw rotation
            euler[0] = 0  # roll = 0
            euler[1] = 0  # pitch = 0
            # euler[2] remains as yaw

            # Convert back to quaternion
            from mujoco import mju_euler2Quat
            new_quat = np.zeros(4)
            mju_euler2Quat(new_quat, euler)

            self.data.xquat[body_id] = new_quat

    def actuate_orient_vertical(self, action: OrientVertical) -> None:
        """Orient agent vertically."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Set quaternion to identity (no rotation)
            self.data.xquat[body_id] = np.array([1.0, 0.0, 0.0, 0.0])

    def actuate_set_yaw(self, action: SetYaw) -> None:
        """Set agent yaw to specific angle."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Create quaternion from yaw angle
            yaw_rad = np.radians(action.rotation_degrees)
            sin_half = np.sin(yaw_rad / 2)
            cos_half = np.cos(yaw_rad / 2)

            # Yaw rotation around Y axis
            yaw_quat = np.array([cos_half, 0, sin_half, 0])
            self.data.xquat[body_id] = yaw_quat

    def actuate_set_agent_pitch(self, action: SetAgentPitch) -> None:
        """Set agent pitch to specific angle."""
        agent = self.get_agent(action.agent_id)
        body_id = agent._body_id

        if body_id is not None:
            # Create quaternion from pitch angle
            pitch_rad = np.radians(action.rotation_degrees)
            sin_half = np.sin(pitch_rad / 2)
            cos_half = np.cos(pitch_rad / 2)

            # Pitch rotation around X axis
            pitch_quat = np.array([cos_half, sin_half, 0, 0])
            self.data.xquat[body_id] = pitch_quat

    def actuate_set_sensor_pitch(self, action: SetSensorPitch) -> None:
        """Set sensor pitch to specific angle."""
        agent = self.get_agent(action.agent_id)

        sensor_id = getattr(action, "sensor_id", None)
        if sensor_id:
            for sensor in agent.sensors:
                if sensor.sensor_id == sensor_id and sensor._camera_id is not None:
                    camera_id = sensor._camera_id

                    # Create quaternion from pitch angle
                    pitch_rad = np.radians(action.rotation_degrees)
                    sin_half = np.sin(pitch_rad / 2)
                    cos_half = np.cos(pitch_rad / 2)

                    # Pitch rotation around X axis
                    pitch_quat = np.array([cos_half, sin_half, 0, 0])

                    # Update camera orientation
                    # Note: This assumes cam_quat exists in the model
                    if hasattr(self.model, "cam_quat"):
                        self.model.cam_quat[camera_id] = pitch_quat
                    break
