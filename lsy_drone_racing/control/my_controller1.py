"""This module implements an AttitudeController for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The attitude control is handled by computing a
PID control law for position tracking, incorporating gravity compensation in thrust calculations.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class AttitudeController(Controller):
    """Example of a controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # For more info on the models, check out https://github.com/utiasDSL/drone-models
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]

        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.3, 0.3, 0.6])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([1.0, 1.0, 0.2])
        self.i_error = np.zeros(3)
        self.g = 9.81

        self._prev_gates = obs["gates_pos"].copy()
        self.create_trajectory(obs)
        self._t_total = 15  # s

        self._tick = 0
        self._finished = False

    def generate_waypoints(
        self, obs: dict[str, NDArray[np.floating]], start_pos: np.arrray
    ) -> NDArray[np.floating]:
        """Compute the waypoints.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            start_pos: The starting position of the drone to generate waypoints at the beginning 
                of the race.

        Returns:
            The waypoints for the trajectory calculation as a numpy array.
        """
        gates_pos = obs["gates_pos"]
        gates_quat = obs["gates_quat"]

        waypoints = [start_pos.copy()]

        for i in range(len(gates_pos)):
            pos = gates_pos[i]
            quat = gates_quat[i]

            forward = R.from_quat(quat).apply([1, 0, 0])

            # approach / center / exit
            before = pos - 0.45 * forward
            center = pos
            after = pos + 0.45 * forward

            waypoints.extend([before, center, after])

        # punto final (hover después del último gate)
        last_forward = R.from_quat(gates_quat[-1]).apply([1, 0, 0])
        finish = gates_pos[-1] + last_forward * 0.7
        waypoints.append(finish)

        return np.array(waypoints)

    def avoid_obstacles(
        self, obs: dict[str, NDArray[np.floating]], waypoints: np.array
    ) -> NDArray[np.floating]:
        """Function to move waypoints to better avoid obstacles.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            waypoints: the current waypoints as a numpy array.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        obstacles = np.asarray(obs["obstacles_pos"], dtype=float)
        clearance = 0.3       # obstacle safety radius
        detour_offset = 0.4    # how far to route around obstacle
        
        pre_turn_dist = 0.6        # begin turning before obstacle

        lookahead_pts = 3          # inspect next 3 segments

        new_wp = [waypoints[0]]

        for i in range(1, len(waypoints)):
            p0 = new_wp[-1]
            p1 = waypoints[i]
            inserted_detour = False
            for j in range(i, min(i + lookahead_pts, len(waypoints)-1)):
                s0 = waypoints[j]
                s1 = waypoints[j+1]
                
                seg = p1 - p0
                seg_len = np.linalg.norm(seg)

                if seg_len < 1e-6:
                    continue

                seg_dir = seg / seg_len
                for obst in obstacles:
                    # Project obstacle onto segment
                    t = np.dot(obst - s0, seg_dir)
                    t = np.clip(t, 0, seg_len)

                    closest = s0 + t * seg_dir
                    dist = np.linalg.norm(obst - closest)

                    if dist < clearance:
                        # Perpendicular direction (2D x-y avoidance)
                        perp = np.array([-seg_dir[1], seg_dir[0], 0.0])

                        # Choose side that moves away from obstacle
                        if np.dot(perp, obst - closest) > 0:
                            perp = -perp

                        detour_wp = (
                        closest
                        - seg_dir * pre_turn_dist
                        + perp * detour_offset)

                        # Insert new waypoint before final waypoint
                        new_wp.append(detour_wp)
                        inserted_detour = True
                        break
                if inserted_detour:
                    break
            new_wp.append(p1)

        return np.array(new_wp)

    def create_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        start = obs["pos"]
        waypoints = self.generate_waypoints(obs, start)

        waypoints = self.avoid_obstacles(obs, waypoints)

        t = np.linspace(0, 20, waypoints.shape[0])

        self._t_total = 20
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()

        

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        if not np.allclose(obs["gates_pos"], self._prev_gates, atol=0.05):
            self._prev_gates = obs["gates_pos"].copy()
            self.create_trajectory(obs)

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * (1 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust[2] += self.drone_mass * self.g

        # Update z_axis to the current orientation of the drone
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]

        # update current thrust
        thrust_desired = target_thrust.dot(z_axis)

        # update z_axis_desired
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        action = np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize the desired trajectory and the current setpoint."""
        setpoint = self._des_pos_spline(self._tick / self._freq).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        trajectory = self._des_pos_spline(np.linspace(0, self._t_total, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
