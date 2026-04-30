"""Structured attitude controller.

Main idea:
- Build one local spline per target gate/sector.
- Use a general PID to track the spline.
- Convert desired force into roll, pitch, yaw, thrust.
- Avoid copying the full hardcoded solution, but keep a reasonable racing-line structure.

This controller is for:

    control_mode = "attitude"

It returns:

    [roll_des, pitch_des, yaw_des, thrust_des]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class AttitudeController(Controller):
    """Spline planner + general PID + attitude/thrust output."""

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
        self._tick = 0
        self._finished = False
        self._t_total = 25.0

        # Drone parameters
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = float(np.asarray(drone_params["mass"]).squeeze())
        self.g = 9.81

        # General PID gains
        self.kp = np.array([0.52, 0.52, 1.45], dtype=float)
        self.ki = np.array([0.035, 0.035, 0.045], dtype=float)
        self.kd = np.array([0.38, 0.38, 0.52], dtype=float)

        self.ki_range = np.array([1.5, 1.5, 0.35], dtype=float)
        self.i_error = np.zeros(3, dtype=float)

        # Automatic trajectory timing
        self.cruise_speed = 0.80
        self.min_sector_time = 1.25
        self.max_sector_time = 3.8

        # Gate passing geometry
        self.default_gate_exit = 0.35
        self.default_gate_approach = 0.25

        # Obstacle avoidance
        self.use_obstacle_avoidance = True
        self.obstacle_trigger_distance = 0.16
        self.obstacle_push_distance = 0.18
        self.max_avoidance_recursion = 3
        self.current_rec_depth = 0

        # Read nominal track from TOML config
        track_gates = list(config.env.track.gates)
        track_obstacles = list(config.env.track.obstacles)
        track_drones = list(config.env.track.drones)

        self.base_gates_pos = np.array(
            [self._cfg_get(gate, "pos") for gate in track_gates], dtype=float
        )

        self.base_gates_rpy = np.array(
            [self._cfg_get(gate, "rpy") for gate in track_gates], dtype=float
        )

        self.base_obstacles_pos = np.array(
            [self._cfg_get(obst, "pos") for obst in track_obstacles], dtype=float
        )

        self.start_pos = np.array(self._cfg_get(track_drones[0], "pos"), dtype=float)

        # These get updated when the sensor reveals randomized positions.
        self.current_gates_pos = self.base_gates_pos.copy()
        self.current_gates_rpy = self.base_gates_rpy.copy()
        self.current_obstacles_pos = self.base_obstacles_pos.copy()

        # Internal trajectory state
        self._des_pos_spline: CubicSpline | None = None
        self._des_vel_spline = None

        self.current_sector = -1
        self._old_gate_index = -1
        self._sector_start_time = 0.0
        self._sector_duration = 2.5
        self._sector_entry_pos = None

        self._first_iteration = True
        self._prev_action = np.zeros(4, dtype=np.float32)

    # Small utilities
    def _cfg_get(self, obj: dict, key: str) -> np.ndarray:
        """Read from dict-like or ConfigDict-like object."""
        if isinstance(obj, dict):
            return obj[key]
        return getattr(obj, key)

    def _target_gate_index(self, target_gate_obs: np.ndarray) -> int:
        """Convert target_gate observation to plain int."""
        return int(np.asarray(target_gate_obs).squeeze())

    def _safe_action(self, action: np.ndarray) -> np.ndarray:
        """Avoid sending NaNs/Infs to the simulator."""
        action = np.asarray(action, dtype=np.float32)
        return np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

    def _path_length(self, waypoints: np.ndarray) -> float:
        """Total piecewise linear path length."""
        if len(waypoints) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))

    def _compute_duration(self, waypoints: np.ndarray) -> float:
        """Compute sector duration from path length."""
        length = self._path_length(waypoints)
        duration = length / max(self.cruise_speed, 1e-3)
        duration = float(np.clip(duration, self.min_sector_time, self.max_sector_time))
        return duration

    def _time_array_for_waypoints(
        self, waypoints: np.ndarray, t0: float, duration: float
    ) -> np.ndarray:
        """Allocate time proportional to segment length.

        This is better than np.linspace when some waypoints are close and others are far.
        """
        if len(waypoints) < 2:
            return np.array([t0, t0 + duration], dtype=float)

        seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        if cumulative[-1] < 1e-6:
            return np.linspace(t0, t0 + duration, len(waypoints))

        return t0 + duration * cumulative / cumulative[-1]

    # Gate geometry
    def _gate_direction_from_rpy(self, rpy: np.ndarray) -> np.ndarray:
        """Horizontal direction vector through a gate from yaw."""
        yaw = rpy[2]
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        norm = np.linalg.norm(direction[:2])

        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0], dtype=float)

        return direction / norm

    def _gate_prev_next(
        self, gate_pos: np.ndarray, gate_rpy: np.ndarray, offset_prev: float, offset_next: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Point before and after gate along gate yaw direction."""
        direction = self._gate_direction_from_rpy(gate_rpy)

        prev_wp = np.asarray(gate_pos, dtype=float) - offset_prev * direction
        next_wp = np.asarray(gate_pos, dtype=float) + offset_next * direction

        return prev_wp, next_wp

    def _shifted_nominal_point(
        self,
        nominal_point: np.ndarray | list[float],
        gates_pos: np.ndarray,
        nearby_gate_ids: list[int],
    ) -> np.ndarray:
        """Adapt a nominal helper waypoint to randomized gate positions.

        Example:
            The nominal racing point [1.2, -0.25, 1.1] worked for the nominal track.
            If gates 0 and 1 move by +[0.1, 0.0, 0.0], we move that helper point too.
        """
        nominal_point = np.asarray(nominal_point, dtype=float)

        ids = np.asarray(nearby_gate_ids, dtype=int)
        observed = gates_pos[ids]
        nominal = self.base_gates_pos[ids]

        mean_shift = np.mean(observed - nominal, axis=0)
        return nominal_point + mean_shift

    # Waypoint generation
    def _build_sector_waypoints(
        self, sector: int, drone_pos: np.ndarray, gates_pos: np.ndarray, add_waypoint: np.ndarray
    ) -> np.ndarray:
        """Generate waypoints for the current sector.

        Args:
            sector: The current sector of the drone.
            drone_pos: The position of the drone.
            gates_pos: The position of the gates.
            add_waypoint: I have no idea honestly but it is a numpy array.

        Returns:
            The waypoints of the sector as a numpy array.
        """
        if sector == 0:
            # Start from actual current drone position at sector entry.
            if self._sector_entry_pos is None:
                start = np.asarray(drone_pos, dtype=float).copy()
                start[2] = max(start[2], 0.05)
                self._sector_entry_pos = start
            else:
                start = self._sector_entry_pos.copy()

            _, gate0_exit = self._gate_prev_next(
                gates_pos[0], self.current_gates_rpy[0], offset_prev=0.40, offset_next=0.40
            )
            gate0_exit[2] = gates_pos[0][2] + 0.20

            waypoints = [start]

            if add_waypoint is not None:
                waypoints.append(add_waypoint)

            waypoints += [gates_pos[0], gate0_exit]

        elif sector == 1:
            helper_1 = self._shifted_nominal_point(
                [1.20, -0.25, 1.10], gates_pos, nearby_gate_ids=[0, 1]
            )

            helper_after = self._shifted_nominal_point(
                [-0.50, -0.05, 0.80], gates_pos, nearby_gate_ids=[1, 2]
            )

            waypoints = [gates_pos[0], helper_1]

            if add_waypoint is not None:
                waypoints.append(add_waypoint)

            waypoints += [gates_pos[1], helper_after]

        elif sector == 2:
            _, gate1_exit = self._gate_prev_next(
                gates_pos[1], self.current_gates_rpy[1], offset_prev=0.10, offset_next=0.20
            )

            _, gate2_exit = self._gate_prev_next(
                gates_pos[2], self.current_gates_rpy[2], offset_prev=0.40, offset_next=0.30
            )

            helper_mid = self._shifted_nominal_point(
                [-0.50, -0.05, 0.80], gates_pos, nearby_gate_ids=[1, 2]
            )

            waypoints = [gate1_exit]

            if add_waypoint is not None:
                waypoints.append(add_waypoint)

            waypoints += [helper_mid, gates_pos[2], gate2_exit]

        elif sector == 3:
            _, gate3_exit = self._gate_prev_next(
                gates_pos[3], self.current_gates_rpy[3], offset_prev=0.20, offset_next=0.40
            )

            helper_mid = self._shifted_nominal_point(
                [-0.50, -0.40, 0.90], gates_pos, nearby_gate_ids=[2, 3]
            )

            waypoints = waypoints = [gates_pos[2], helper_mid]

            if add_waypoint is not None:
                waypoints.append(add_waypoint)

            waypoints += [gates_pos[3], gate3_exit]

        else:
            # Fallback hover if something unexpected happens.
            waypoints = [drone_pos, drone_pos]

        return np.array(waypoints, dtype=float)

    # Obstacle avoidance
    def _check_obstacle_collision(
        self, spline: CubicSpline, sector: int, obstacle_index: int
    ) -> np.ndarray | None:
        """Check if current spline passes too close to one obstacle in XY.

        Args:
            spline: The current spline.
            sector: The current sector.
            obstacle_index: The index of the next obstacle.

        Returns:
            A waypoint to insert if too close.
        """
        if not self.use_obstacle_avoidance:
            return None

        if obstacle_index < 0 or obstacle_index >= len(self.current_obstacles_pos):
            return None

        t0 = spline.x[0]
        tf = spline.x[-1]
        t_values = np.linspace(t0, tf, 40)

        obstacle = self.current_obstacles_pos[obstacle_index]

        min_dist = 1e9
        closest_delta = None
        closest_pos = None

        for ti in t_values:
            des_pos = spline(ti)
            delta = obstacle[:2] - des_pos[:2]
            dist = np.linalg.norm(delta)

            if dist < min_dist:
                min_dist = dist
                closest_delta = delta
                closest_pos = des_pos

        if min_dist >= self.obstacle_trigger_distance:
            return None

        if closest_delta is None or closest_pos is None:
            return None

        push = (closest_delta / (min_dist + 1e-3)) * self.obstacle_push_distance

        avoidance_wp = np.array(
            [obstacle[0] - push[0], obstacle[1] - push[1], closest_pos[2]], dtype=float
        )

        return avoidance_wp

    # Spline construction
    def _build_sector_spline(
        self,
        t_now: float,
        sector: int,
        drone_pos: np.ndarray,
        gates_pos: np.ndarray,
        add_waypoint: np.ndarray | None = None,
    ) -> CubicSpline:
        """Build the current sector spline.

        Args:
            t_now: The current timestep.
            sector: The current sector.
            drone_pos: The position before the new spline.
            gates_pos: The position of the gates for this sector.
            add_waypoint: Something something numpy array.

        Returns:
            The spline that the drone will use in the next sector.
        """
        waypoints = self._build_sector_waypoints(
            sector=sector, drone_pos=drone_pos, gates_pos=gates_pos, add_waypoint=add_waypoint
        )

        duration = self._compute_duration(waypoints)
        self._sector_duration = duration

        t_array = self._time_array_for_waypoints(
            waypoints=waypoints, t0=self._sector_start_time, duration=duration
        )

        spline = CubicSpline(t_array, waypoints)

        # Obstacle correction: check obstacle with same index as sector.
        avoidance_wp = self._check_obstacle_collision(
            spline=spline, sector=sector, obstacle_index=sector
        )

        if avoidance_wp is not None and self.current_rec_depth < self.max_avoidance_recursion:
            self.current_rec_depth += 1

            spline = self._build_sector_spline(
                t_now=t_now,
                sector=sector,
                drone_pos=drone_pos,
                gates_pos=gates_pos,
                add_waypoint=avoidance_wp,
            )

            self.current_rec_depth = 0

        return spline

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

        if t >= self._t_total:
            self._finished = True

        target_gate = self._target_gate_index(obs["target_gate"])

        if target_gate == -1:
            self._finished = True
            return self._prev_action

        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)

        drone_pos = np.asarray(obs["pos"], dtype=float)
        drone_vel = np.asarray(obs["vel"], dtype=float)
        drone_quat = np.asarray(obs["quat"], dtype=float)

        # Convert observed gate orientations to RPY.
        observed_gates_rpy = R.from_quat(gates_quat).as_euler("xyz", degrees=False)

        # Detect sector change.
        changed_sector = target_gate != self.current_sector

        if changed_sector:
            self.current_sector = target_gate
            self.i_error = np.zeros(3, dtype=float)
            self._sector_start_time = t

            self._sector_entry_pos = drone_pos.copy()
            self._sector_entry_pos[2] = max(self._sector_entry_pos[2], 0.05)

        # Detect sensor updates from level 2.
        gates_changed = not np.allclose(gates_pos, self.current_gates_pos, atol=1e-3)
        gate_rpy_changed = not np.allclose(observed_gates_rpy, self.current_gates_rpy, atol=1e-3)
        obstacles_changed = not np.allclose(obstacles_pos, self.current_obstacles_pos, atol=1e-3)

        should_replan = (
            self._first_iteration
            or changed_sector
            or gates_changed
            or gate_rpy_changed
            or obstacles_changed
            or self._des_pos_spline is None
        )

        if should_replan:
            self.current_gates_pos = gates_pos.copy()
            self.current_gates_rpy = observed_gates_rpy.copy()
            self.current_obstacles_pos = obstacles_pos.copy()

            self._des_pos_spline = self._build_sector_spline(
                t_now=t, sector=target_gate, drone_pos=drone_pos, gates_pos=gates_pos
            )

            self._des_vel_spline = self._des_pos_spline.derivative()
            self._first_iteration = False

        action = self._pid_track_spline(
            spline=self._des_pos_spline,
            vel_spline=self._des_vel_spline,
            drone_pos=drone_pos,
            drone_vel=drone_vel,
            drone_quat=drone_quat,
            t=t,
        )

        action = self._safe_action(action)
        self._prev_action = action

        return action

    # PID tracking
    def _pid_track_spline(
        self,
        spline: CubicSpline,
        vel_spline: CubicSpline,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_quat: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Track the spline with a general PID and output attitude command.

        Args:
            spline: The spline that the drone follows.
            vel_spline: The velocity derived from that spline.
            drone_pos: The position of the drone.
            drone_vel: The velocity of the drone.
            drone_quat: The inclination of the drone.
            t: The current time.

        Returns:
            The action for the pid as a numpy array.
        """
        t_eval = float(np.clip(t, spline.x[0], spline.x[-1]))

        des_pos = spline(t_eval)
        des_vel = vel_spline(t_eval)
        des_yaw = 0.0

        pos_error = des_pos - drone_pos
        vel_error = des_vel - drone_vel

        self.i_error += pos_error * (1.0 / self._freq)
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        target_force = np.zeros(3, dtype=float)
        target_force += self.kp * pos_error
        target_force += self.ki * self.i_error
        target_force += self.kd * vel_error

        # Gravity compensation.
        target_force[2] += self.drone_mass * self.g

        force_norm = np.linalg.norm(target_force)

        if force_norm < 1e-6:
            target_force = np.array([0.0, 0.0, self.drone_mass * self.g], dtype=float)
            force_norm = np.linalg.norm(target_force)

        # Current body z-axis in world frame.
        z_axis_current = R.from_quat(drone_quat).as_matrix()[:, 2]

        # Collective thrust is desired force projected onto current body z-axis.
        thrust_desired = float(np.dot(target_force, z_axis_current))
        thrust_desired = max(0.0, thrust_desired)

        # Desired body z-axis points in direction of desired force.
        z_axis_desired = target_force / force_norm

        x_c_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0], dtype=float)

        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_norm = np.linalg.norm(y_axis_desired)

        if y_norm < 1e-6:
            y_axis_desired = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y_axis_desired /= y_norm

        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)
        x_norm = np.linalg.norm(x_axis_desired)

        if x_norm < 1e-6:
            x_axis_desired = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x_axis_desired /= x_norm

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T

        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)
        euler_desired = np.clip(euler_desired, -np.pi / 2, np.pi / 2)

        action = np.array(
            [euler_desired[0], euler_desired[1], euler_desired[2], thrust_desired], dtype=np.float32
        )

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
        self._tick = 0
        self._finished = False
        self._first_iteration = True

        self._des_pos_spline = None
        self._des_vel_spline = None

        self.current_sector = -1
        self._old_gate_index = -1
        self._sector_start_time = 0.0
        self._sector_duration = 2.5
        self._sector_entry_pos = None

        self.current_gates_pos = self.base_gates_pos.copy()
        self.current_gates_rpy = self.base_gates_rpy.copy()
        self.current_obstacles_pos = self.base_obstacles_pos.copy()

        self.i_error = np.zeros(3, dtype=float)
        self.current_rec_depth = 0

        self._prev_action = np.zeros(4, dtype=np.float32)

    def render_callback(self, sim: Sim):
        """Visualize current spline and setpoint."""
        if self._des_pos_spline is None:
            return

        t = self._tick / self._freq
        t_eval = float(np.clip(t, self._des_pos_spline.x[0], self._des_pos_spline.x[-1]))

        setpoint = self._des_pos_spline(t_eval).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)

        t_values = np.linspace(self._des_pos_spline.x[0], self._des_pos_spline.x[-1], 100)
        trajectory = self._des_pos_spline(t_values)
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))
