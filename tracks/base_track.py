"""
Abstract base class for track definitions.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline
from config import (
    IDX_PX, IDX_PY, IDX_PSI, IDX_V, IDX_OMEGA,
    A_LAT_MAX, V_MAX, REF_SPEED_SCALE, ENABLE_OBSTACLES,
)


class BaseTrack(ABC):
    """Abstract base class for racing tracks."""

    def __init__(self):
        self._centerline_x = None
        self._centerline_y = None
        self._heading = None
        self._curvature = None
        self._arc_length = None
        self._obstacles = []
        self._rect_obstacles = []
        self._total_length = 0.0
        self._num_points = 0

    @abstractmethod
    def _build_track(self):
        """Build the track geometry. Must set centerline, heading, curvature."""
        pass

    def get_centerline(self):
        """Returns (x_array, y_array) of the track centerline."""
        return self._centerline_x.copy(), self._centerline_y.copy()

    def get_heading(self):
        """Returns heading angle array along the track."""
        return self._heading.copy()

    def get_curvature(self):
        """Returns curvature array along the track."""
        return self._curvature.copy()

    def get_arc_length(self):
        """Returns cumulative arc length array."""
        return self._arc_length.copy()

    def get_obstacles(self):
        """Returns list of (ox, oy, radius) tuples."""
        if not ENABLE_OBSTACLES:
            return []
        return list(self._obstacles)

    def get_rect_obstacles(self):
        """Returns list of rectangular obstacles as (cx, cy, length, width, angle_rad)."""
        if not ENABLE_OBSTACLES:
            return []
        return list(self._rect_obstacles)

    def total_length(self):
        """Total track length in meters."""
        return self._total_length

    def num_points(self):
        """Number of discretization points."""
        return self._num_points

    def closest_point(self, px, py):
        """
        Find the closest point on the track to (px, py).

        Returns:
            idx: index of closest point
            s: arc length of closest point
            lateral_error: signed lateral distance
        """
        dx = self._centerline_x - px
        dy = self._centerline_y - py
        dist = np.sqrt(dx**2 + dy**2)
        idx = np.argmin(dist)

        # Compute signed lateral error
        heading = self._heading[idx]
        nx = -np.sin(heading)  # Normal vector (pointing left)
        ny = np.cos(heading)
        lateral_error = (px - self._centerline_x[idx]) * nx + \
                        (py - self._centerline_y[idx]) * ny

        return idx, self._arc_length[idx], lateral_error

    def get_reference_trajectory(self, start_idx, horizon, v_ref=None,
                                 a_lat_max=A_LAT_MAX, v_max=V_MAX):
        """
        Generate reference trajectory for MPC.

        Args:
            start_idx: starting index on the track
            horizon: prediction horizon T
            v_ref: reference velocity (if None, computed from curvature)
            a_lat_max: max lateral acceleration for speed profile
            v_max: maximum velocity

        Returns:
            ref: (horizon, 5) reference states [px, py, psi, v, omega]
        """
        ref = np.zeros((horizon, 5))
        N = self._num_points

        for t in range(horizon):
            idx = (start_idx + t) % N
            ref[t, IDX_PX] = self._centerline_x[idx]  # px
            ref[t, IDX_PY] = self._centerline_y[idx]  # py
            ref[t, IDX_PSI] = self._heading[idx]      # psi

            # Speed from curvature
            kappa = abs(self._curvature[idx])
            if v_ref is not None:
                ref[t, IDX_V] = REF_SPEED_SCALE * v_ref
            elif kappa > 1e-6:
                ref[t, IDX_V] = REF_SPEED_SCALE * min(v_max, np.sqrt(a_lat_max / kappa))
            else:
                ref[t, IDX_V] = REF_SPEED_SCALE * v_max

            ref[t, IDX_OMEGA] = ref[t, IDX_V] * self._curvature[idx]  # omega = v * kappa

        return ref

    def get_reference_v_omega(self, start_idx, horizon, a_lat_max=A_LAT_MAX, v_max=V_MAX):
        """
        Get reference [v, omega] trajectory for Koopman MPC cost.

        Returns:
            y_ref: (horizon, 2) reference [v, omega]
        """
        ref = self.get_reference_trajectory(start_idx, horizon,
                                            a_lat_max=a_lat_max, v_max=v_max)
        return ref[:, [IDX_V, IDX_OMEGA]]  # [v, omega]

    def _compute_geometry(self, x, y):
        """Compute heading, curvature, and arc length from (x, y) coordinates."""
        N = len(x)

        # Arc length
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        self._arc_length = np.zeros(N)
        self._arc_length[1:] = np.cumsum(ds)
        self._total_length = self._arc_length[-1]

        # Heading from finite differences
        self._heading = np.arctan2(
            np.gradient(y, self._arc_length, edge_order=2),
            np.gradient(x, self._arc_length, edge_order=2)
        )

        # Curvature: d(heading)/ds
        dheading = np.gradient(self._heading)
        # Handle angle wrapping
        dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
        self._curvature = dheading / np.maximum(np.gradient(self._arc_length), 1e-6)

        self._num_points = N
