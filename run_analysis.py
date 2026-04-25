"""
Entry point: Generate all figures and tables from simulation results.
"""

import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR,
    SIGMA_VALUES, THETA_VALUES, EPSILON_VALUES
)
from simulation.simulator import Simulator
from simulation.metrics import compute_all_metrics
from tracks.lusail_track import LusailTrack
from tracks.custom_track import CustomWindingTrack
from disturbance.disturbance_generator import DisturbanceGenerator
from visualization.plot_trajectories import (
    plot_trajectory_comparison, plot_state_comparison, plot_control_comparison
)
from visualization.plot_tables import (
    print_table_6, print_performance_tables,
    print_robustness_table, print_sensitivity_table
)


def load_results(track_name, methods=None):
    """Load simulation results for a given track."""
    if methods is None:
        methods = ['LMPC', 'NMPC', 'K-MPC', 'K-DRMPC']

    results = {}
    for method in methods:
        filepath = os.path.join(RESULTS_DIR, f"{method}_{track_name}.pkl")
        if os.path.exists(filepath):
            results[method] = Simulator.load_result(filepath)
        else:
            print(f"  Warning: Result not found for {method} on {track_name}")

    return results


def analyze_track(track, track_name, results, w_samples=None, C_matrix=None):
    """Compute metrics and generate plots for a track."""
    methods = list(results.keys())
    obstacles = track.get_obstacles()

    # Compute metrics for all methods
    all_metrics = {}
    for method, result in results.items():
        print(f"  Computing metrics for {method}...")
        metrics = compute_all_metrics(
            result, track, obstacles,
            w_samples=w_samples, C_matrix=C_matrix
        )
        all_metrics[method] = metrics

    return all_metrics


def main():
    print("=" * 60)
    print("Analysis and Visualization Pipeline")
    print("=" * 60)

    # Create tracks
    lusail = LusailTrack()
    custom = CustomWindingTrack()

    # Load disturbance info
    dist_gen = DisturbanceGenerator(sigma=0.05)
    w_samples = dist_gen.get_empirical_samples(100)

    # Load Koopman matrices
    matrices_path = os.path.join(MODEL_DIR, 'koopman_matrices.npz')
    C_matrix = None
    if os.path.exists(matrices_path):
        mats = np.load(matrices_path)
        C_matrix = mats['C']

    # Load training log for Table 6
    log_path = os.path.join(MODEL_DIR, 'training_log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            training_log = json.load(f)
        print_table_6(training_log)

    # ===== Lusail Circuit Analysis =====
    print("\n" + "#" * 60)
    print("# LUSAIL CIRCUIT ANALYSIS")
    print("#" * 60)

    lusail_results = load_results('LusailTrack')
    if lusail_results:
        lusail_metrics = analyze_track(
            lusail, 'LusailTrack', lusail_results,
            w_samples=w_samples, C_matrix=C_matrix
        )

        # Table 9: Performance comparison
        methods = ['LMPC', 'NMPC', 'K-MPC', 'K-DRMPC']
        avail_methods = [m for m in methods if m in lusail_metrics]
        print_performance_tables(lusail_metrics, avail_methods,
                                'Lusail Circuit', table_num="9")

        # Table 10: Safety performance
        print_performance_tables(lusail_metrics, avail_methods,
                                'Lusail Circuit (Safety)', table_num="10")

        # Table 12: Computation time
        print(f"\nTable 12: Computation Time Comparison")
        print("-" * 60)
        for m in avail_methods:
            mean_t = lusail_metrics[m].get('solve_time_mean', 0)
            max_t = lusail_metrics[m].get('solve_time_max', 0)
            feasible = lusail_metrics[m].get('real_time_feasible', False)
            print(f"  {m:<12}: mean={mean_t:.1f}ms, max={max_t:.1f}ms, "
                  f"RT feasible={'Yes' if feasible else 'No'}")

        # Figure 6: Trajectories
        plot_trajectory_comparison(
            lusail_results, lusail,
            title="Vehicle Trajectories on Lusail Circuit",
            filename="fig6_lusail_trajectory.pdf"
        )

        # State evolution plots
        plot_state_comparison(lusail_results, lusail,
                            filename="lusail_states.pdf")
    else:
        print("  No Lusail results found.")

    # ===== Custom Winding Track Analysis =====
    print("\n" + "#" * 60)
    print("# CUSTOM WINDING TRACK ANALYSIS")
    print("#" * 60)

    custom_results = load_results('CustomWindingTrack')
    if custom_results:
        custom_metrics = analyze_track(
            custom, 'CustomWindingTrack', custom_results,
            w_samples=w_samples, C_matrix=C_matrix
        )

        # Table 13: Performance on custom track
        methods = ['LMPC', 'NMPC', 'K-MPC', 'K-DRMPC']
        avail_methods = [m for m in methods if m in custom_metrics]
        print_performance_tables(custom_metrics, avail_methods,
                                'Custom Winding Track', table_num="13")

        # Figure 7: Trajectories
        plot_trajectory_comparison(
            custom_results, custom,
            title="Vehicle Trajectories on Custom Winding Track",
            filename="fig7_custom_trajectory.pdf"
        )
    else:
        print("  No Custom Winding Track results found.")

    # ===== Robustness Analysis (Table 11) =====
    print("\n" + "#" * 60)
    print("# ROBUSTNESS ANALYSIS")
    print("#" * 60)

    robustness_results = {}
    for sigma in SIGMA_VALUES:
        key = f"sigma_{sigma}"
        filepath = os.path.join(RESULTS_DIR, f"robustness_{key}.pkl")
        if os.path.exists(filepath):
            result = Simulator.load_result(filepath)
            metrics = compute_all_metrics(result, lusail,
                                         w_samples=w_samples,
                                         C_matrix=C_matrix)
            robustness_results[key] = metrics

    if robustness_results:
        print_robustness_table(robustness_results, SIGMA_VALUES, table_num="11")

    # ===== Sensitivity Analysis - Theta (Table 14) =====
    print("\n" + "#" * 60)
    print("# SENSITIVITY ANALYSIS - THETA")
    print("#" * 60)

    theta_results = {}
    for theta in THETA_VALUES:
        key = f"theta_{theta}"
        filepath = os.path.join(RESULTS_DIR, f"sensitivity_{key}.pkl")
        if os.path.exists(filepath):
            result = Simulator.load_result(filepath)
            metrics = compute_all_metrics(result, lusail,
                                         w_samples=w_samples,
                                         C_matrix=C_matrix)
            theta_results[key] = metrics

    if theta_results:
        print_sensitivity_table(theta_results, THETA_VALUES, "theta",
                               table_num="14")

    # ===== Sensitivity Analysis - Epsilon (Table 15) =====
    print("\n" + "#" * 60)
    print("# SENSITIVITY ANALYSIS - EPSILON")
    print("#" * 60)

    epsilon_results = {}
    for epsilon in EPSILON_VALUES:
        key = f"epsilon_{epsilon}"
        filepath = os.path.join(RESULTS_DIR, f"sensitivity_{key}.pkl")
        if os.path.exists(filepath):
            result = Simulator.load_result(filepath)
            metrics = compute_all_metrics(result, lusail,
                                         w_samples=w_samples,
                                         C_matrix=C_matrix)
            epsilon_results[key] = metrics

    if epsilon_results:
        print_sensitivity_table(epsilon_results, EPSILON_VALUES, "epsilon",
                               table_num="15")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
