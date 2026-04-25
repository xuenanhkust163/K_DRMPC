"""
Trajectory visualization for Figures 6 and 7.
Overlays all 4 methods' trajectories on the track with obstacles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    METHOD_COLORS, METHOD_LABELS, FIGURES_DIR,
    FIGURE_DPI, FIGURE_FORMAT, D_SAFE, VEHICLE_RADIUS
)


def plot_trajectory_comparison(results, track, title="Trajectory Comparison",
                               filename=None, save_dir=FIGURES_DIR):
    """
    Plot all methods' trajectories on a single figure.

    Args:
        results: dict {method_name: SimResult}
        track: BaseTrack instance
        title: figure title
        filename: output filename
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Plot track centerline
    cx, cy = track.get_centerline()
    ax.plot(cx, cy, '--', color='gray', linewidth=1.5, alpha=0.6, label='Track centerline')

    # Plot obstacles
    obstacles = track.get_obstacles()
    for i, (ox, oy, r) in enumerate(obstacles):
        # Obstacle body
        circle = Circle((ox, oy), r, color='red', alpha=0.4)
        ax.add_patch(circle)
        # Safety margin
        circle_safe = Circle((ox, oy), r + VEHICLE_RADIUS + D_SAFE,
                            fill=False, edgecolor='red', linestyle='--',
                            linewidth=1, alpha=0.5)
        ax.add_patch(circle_safe)
        ax.annotate(f'Obs {i+1}', (ox, oy), fontsize=8, ha='center', va='center')

    # Plot each method's trajectory
    for method_name, result in results.items():
        data = result.to_arrays()
        states = data['states']
        color = METHOD_COLORS.get(method_name, 'black')
        label = METHOD_LABELS.get(method_name, method_name)
        ax.plot(states[:, 0], states[:, 1], color=color, linewidth=1.5,
                label=label, alpha=0.8)

        # Mark start position
        ax.plot(states[0, 0], states[0, 1], 'o', color=color, markersize=8)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is None:
        filename = f"trajectory_{track.__class__.__name__}.{FIGURE_FORMAT}"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Trajectory plot saved to {filepath}")


def plot_state_comparison(results, track, filename=None, save_dir=FIGURES_DIR):
    """
    Plot state variable comparisons over time for all methods.
    """
    os.makedirs(save_dir, exist_ok=True)

    state_labels = ['$p_x$ (m)', '$p_y$ (m)', '$v$ (m/s)',
                    '$\\psi$ (rad)', '$\\omega$ (rad/s)']

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)

    for method_name, result in results.items():
        data = result.to_arrays()
        states = data['states']
        t = np.arange(len(states)) * 0.1  # DT = 0.1s
        color = METHOD_COLORS.get(method_name, 'black')
        label = METHOD_LABELS.get(method_name, method_name)

        for i in range(5):
            axes[i].plot(t, states[:, i], color=color, linewidth=1,
                        label=label, alpha=0.8)

    for i in range(5):
        axes[i].set_ylabel(state_labels[i], fontsize=11)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='best', fontsize=9)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle(f'State Evolution: {track.__class__.__name__}', fontsize=14)

    plt.tight_layout()

    if filename is None:
        filename = f"states_{track.__class__.__name__}.{FIGURE_FORMAT}"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"State comparison plot saved to {filepath}")


def plot_control_comparison(results, filename=None, save_dir=FIGURES_DIR):
    """
    Plot control inputs over time for all methods.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    control_labels = ['Acceleration $a$ (m/s$^2$)', 'Steering $\\delta$ (rad)']

    for method_name, result in results.items():
        data = result.to_arrays()
        controls = data['controls']
        t = np.arange(len(controls)) * 0.1
        color = METHOD_COLORS.get(method_name, 'black')
        label = METHOD_LABELS.get(method_name, method_name)

        for i in range(2):
            axes[i].plot(t, controls[:, i], color=color, linewidth=1,
                        label=label, alpha=0.8)

    for i in range(2):
        axes[i].set_ylabel(control_labels[i], fontsize=11)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='best', fontsize=9)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle('Control Input Comparison', fontsize=14)

    plt.tight_layout()

    if filename is None:
        filename = f"controls.{FIGURE_FORMAT}"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Control comparison plot saved to {filepath}")
