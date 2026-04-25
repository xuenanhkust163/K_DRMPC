"""
Custom winding track approximation (~3.2 km) with S-curves and chicanes.
3 static obstacles placed at apex points.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
from config import OBSTACLE_RADIUS


class CustomWindingTrack(BaseTrack):
    """
    Custom winding track with aggressive S-curves.
    ~3.2 km length, 3 static obstacles at apex points.
    """

    def __init__(self, num_points=1500):
        super().__init__()
        self._target_points = num_points
        self._build_track()

    def _build_track(self):
        """Build a winding track using parametric sinusoidal curves."""
        # Create a closed track using a combination of circular arc
        # and sinusoidal perturbations
        n_build = 5000  # High-res for initial construction

        # Base shape: elongated oval
        t = np.linspace(0, 2 * np.pi, n_build, endpoint=False)

        # Oval base
        a_major = 500   # Semi-major axis
        b_minor = 200   # Semi-minor axis

        # Add sinusoidal perturbations for winding sections
        r_base = np.sqrt((a_major * np.cos(t))**2 + (b_minor * np.sin(t))**2)

        # Perturbation: add S-curves
        perturbation = (60 * np.sin(3 * t) +
                        40 * np.sin(5 * t) +
                        25 * np.sin(7 * t))

        x_raw = (a_major + perturbation * 0.3) * np.cos(t)
        y_raw = (b_minor + perturbation * 0.5) * np.sin(t)

        # Compute raw length and scale to ~3.2 km
        dx = np.diff(np.append(x_raw, x_raw[0]))
        dy = np.diff(np.append(y_raw, y_raw[0]))
        ds = np.sqrt(dx**2 + dy**2)
        raw_length = np.sum(ds)
        scale = 3200.0 / raw_length

        x_raw *= scale
        y_raw *= scale

        # Fit periodic cubic spline for smooth track
        t_param = np.zeros(n_build + 1)
        for i in range(n_build):
            t_param[i+1] = t_param[i] + np.sqrt(
                (x_raw[(i+1) % n_build] - x_raw[i])**2 +
                (y_raw[(i+1) % n_build] - y_raw[i])**2
            )
        t_param_norm = t_param[:-1] / t_param[-1]

        # Add wrap-around point for periodicity
        x_ext = np.append(x_raw, x_raw[0])
        y_ext = np.append(y_raw, y_raw[0])
        t_ext = np.append(t_param_norm, 1.0)

        cs_x = CubicSpline(t_ext, x_ext, bc_type='periodic')
        cs_y = CubicSpline(t_ext, y_ext, bc_type='periodic')

        # Resample uniformly
        t_fine = np.linspace(0, 1, self._target_points, endpoint=False)
        self._centerline_x = cs_x(t_fine)
        self._centerline_y = cs_y(t_fine)

        # Compute geometry
        self._compute_geometry(self._centerline_x, self._centerline_y)

        # Place 3 obstacles at apex points
        self._place_obstacles()

        print(f"Custom Winding Track: {self._total_length:.0f}m, "
              f"{self._num_points} points, {len(self._obstacles)} obstacles")

    def _place_obstacles(self):
        """Place 3 static obstacles at the tightest curve apexes."""
        N = self._num_points
        curvature = np.abs(self._curvature)

        from scipy.ndimage import uniform_filter1d
        smooth_curv = uniform_filter1d(curvature, size=N // 15, mode='wrap')

        corner_indices = []
        min_separation = N // 6

        for _ in range(3):
            idx = np.argmax(smooth_curv)
            corner_indices.append(idx)
            start = max(0, idx - min_separation)
            end = min(N, idx + min_separation)
            smooth_curv[start:end] = 0

        corner_indices.sort()

        offset_distance = 4.0
        for idx in corner_indices:
            heading = self._heading[idx]
            sign = np.sign(self._curvature[idx])
            nx = -np.sin(heading) * sign
            ny = np.cos(heading) * sign
            ox = self._centerline_x[idx] + offset_distance * nx
            oy = self._centerline_y[idx] + offset_distance * ny
            self._obstacles.append((ox, oy, OBSTACLE_RADIUS))
