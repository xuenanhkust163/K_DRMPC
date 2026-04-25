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
from tracks.lusail_short_track import LusailShortTrack
from tracks.custom_track import CustomWindingTrack
from disturbance.disturbance_generator import DisturbanceGenerator
from visualization.plot_trajectories import (
    plot_trajectory_comparison, plot_state_comparison, plot_control_comparison
)
from visualization.plot_tables import (
    print_table_6, print_performance_tables,
    print_robustness_table, print_sensitivity_table
)


def export_result_to_step_log(result, output_path):
    """Export a simulation result to a readable step-by-step text log."""
    data = result.to_arrays()
    states = data['states']
    controls = data['controls']
    solve_times = data['solve_times']
    timestamps = data['timestamps']
    ref_states = np.array(result.ref_states)
    solve_statuses = list(result.solve_statuses)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f"method={result.method_name}\n")
        f.write(f"track={result.track_name}\n")
        f.write(f"lap_completed={result.lap_completed}\n")
        f.write(f"lap_time={result.lap_time}\n")
        f.write(f"total_steps={result.total_steps}\n")
        f.write("\n")

        if len(states) > 0:
            f.write(
                "init "
                f"x=[{states[0, 0]:.6f}, {states[0, 1]:.6f}, {states[0, 2]:.6f}, {states[0, 3]:.6f}, {states[0, 4]:.6f}]\n"
            )

        for step in range(len(controls)):
            x_t = states[step]
            x_next = states[step + 1]
            u_t = controls[step]
            ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
            solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
            status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
            t_sim = timestamps[step] if step < len(timestamps) else float(step)

            f.write(
                f"step={step:04d} t={t_sim:8.3f}s "
                f"status={status} solve_ms={solve_time_ms:8.3f} "
                f"x=[{x_t[0]:.6f}, {x_t[1]:.6f}, {x_t[2]:.6f}, {x_t[3]:.6f}, {x_t[4]:.6f}] "
                f"ref=[{ref_t[0]:.6f}, {ref_t[1]:.6f}, {ref_t[2]:.6f}, {ref_t[3]:.6f}, {ref_t[4]:.6f}] "
                f"u=[{u_t[0]:.6f}, {u_t[1]:.6f}] "
                f"x_next=[{x_next[0]:.6f}, {x_next[1]:.6f}, {x_next[2]:.6f}, {x_next[3]:.6f}, {x_next[4]:.6f}]\n"
            )


def export_result_to_compact_log(result, output_path):
    """Export a compact one-line-per-step log for quick inspection."""
    data = result.to_arrays()
    states = data['states']
    controls = data['controls']
    solve_times = data['solve_times']
    timestamps = data['timestamps']
    ref_states = np.array(result.ref_states)
    solve_statuses = list(result.solve_statuses)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# step t(s) v ref_v omega ref_omega a delta solve_ms status\n")

        for step in range(len(controls)):
            x_t = states[step]
            u_t = controls[step]
            ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
            solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
            status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
            t_sim = timestamps[step] if step < len(timestamps) else float(step)

            f.write(
                f"{step:04d} "
                f"{t_sim:8.3f} "
                f"{x_t[3]:9.4f} "
                f"{ref_t[3]:9.4f} "
                f"{x_t[4]:9.4f} "
                f"{ref_t[4]:9.4f} "
                f"{u_t[0]:9.4f} "
                f"{u_t[1]:9.4f} "
                f"{solve_time_ms:9.3f} "
                f"{status}\n"
            )


def export_all_result_logs(results_dir=RESULTS_DIR):
    """Export every simulation pkl in the results directory to a text log."""
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    pkl_paths = sorted(glob.glob(os.path.join(results_dir, '*.pkl')))
    for pkl_path in pkl_paths:
        result = Simulator.load_result(pkl_path)
        base_name = os.path.splitext(os.path.basename(pkl_path))[0]
        log_path = os.path.join(log_dir, f"{base_name}.log")
        compact_log_path = os.path.join(log_dir, f"{base_name}.compact.log")
        export_result_to_step_log(result, log_path)
        export_result_to_compact_log(result, compact_log_path)
        print(f"  Step log exported: {log_path}")
        print(f"  Compact log exported: {compact_log_path}")


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


def analyze_track(track, track_name, results, w_samples=None):
    """Compute metrics and generate plots for a track."""
    methods = list(results.keys())
    obstacles = track.get_obstacles()

    # Compute metrics for all methods
    all_metrics = {}
    for method, result in results.items():
        print(f"  Computing metrics for {method}...")
        metrics = compute_all_metrics(
            result, track, obstacles,
            w_samples=w_samples
        )
        all_metrics[method] = metrics

    return all_metrics


def main():
    print("=" * 60)
    print("Analysis and Visualization Pipeline")
    print("=" * 60)

    print("\n--- Exporting result logs ---")
    export_all_result_logs()

    # Create tracks
    lusail_short = LusailShortTrack()
    lusail = LusailTrack()
    custom = CustomWindingTrack()

    # Load disturbance info
    dist_gen = DisturbanceGenerator(sigma=0.05)
    w_samples = dist_gen.get_empirical_samples(100)

    # Load training log for Table 6
    log_path = os.path.join(MODEL_DIR, 'training_log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            training_log = json.load(f)
        print_table_6(training_log)

    # ===== Lusail Short Circuit Analysis =====
    print("\n" + "#" * 60)
    print("# LUSAIL SHORT CIRCUIT ANALYSIS")
    print("#" * 60)

    lusail_short_results = load_results('LusailShortTrack')
    if lusail_short_results:
        lusail_short_metrics = analyze_track(
            lusail_short, 'LusailShortTrack', lusail_short_results,
            w_samples=w_samples
        )

        # Table 9: Performance comparison
        methods = ['LMPC', 'NMPC', 'K-MPC', 'K-DRMPC']
        avail_methods = [m for m in methods if m in lusail_short_metrics]
        print_performance_tables(lusail_short_metrics, avail_methods,
                                'Lusail Short Circuit', table_num="9")

        # Table 10: Safety performance
        print_performance_tables(lusail_short_metrics, avail_methods,
                                'Lusail Short Circuit (Safety)', table_num="10")

        # Table 12: Computation time
        print(f"\nTable 12: Computation Time Comparison")
        print("-" * 60)
        for m in avail_methods:
            mean_t = lusail_short_metrics[m].get('solve_time_mean', 0)
            max_t = lusail_short_metrics[m].get('solve_time_max', 0)
            feasible = lusail_short_metrics[m].get('real_time_feasible', False)
            print(f"  {m:<12}: mean={mean_t:.1f}ms, max={max_t:.1f}ms, "
                  f"RT feasible={'Yes' if feasible else 'No'}")

        # Figure 6: Trajectories
        plot_trajectory_comparison(
            lusail_short_results, lusail_short,
            title="Vehicle Trajectories on Lusail Short Circuit",
            filename="fig6_lusail_short_trajectory.pdf"
        )

        # State evolution plots
        plot_state_comparison(lusail_short_results, lusail_short,
                            filename="lusail_short_states.pdf")
    else:
        print("  No LusailShortTrack results found.")

    # Optional: Original Lusail analysis if historical pkl exists
    print("\n" + "#" * 60)
    print("# LUSAIL CIRCUIT ANALYSIS (OPTIONAL)")
    print("#" * 60)
    lusail_results = load_results('LusailTrack')
    if lusail_results:
        lusail_metrics = analyze_track(
            lusail, 'LusailTrack', lusail_results,
            w_samples=w_samples
        )
        methods = ['LMPC', 'NMPC', 'K-MPC', 'K-DRMPC']
        avail_methods = [m for m in methods if m in lusail_metrics]
        print_performance_tables(lusail_metrics, avail_methods,
                                'Lusail Circuit', table_num="9_lusail")

    # ===== Custom Winding Track Analysis =====
    print("\n" + "#" * 60)
    print("# CUSTOM WINDING TRACK ANALYSIS")
    print("#" * 60)

    custom_results = load_results('CustomWindingTrack')
    if custom_results:
        custom_metrics = analyze_track(
            custom, 'CustomWindingTrack', custom_results,
            w_samples=w_samples
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
            metrics = compute_all_metrics(result, lusail_short,
                                         w_samples=w_samples)
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
            metrics = compute_all_metrics(result, lusail_short,
                                         w_samples=w_samples)
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
            metrics = compute_all_metrics(result, lusail_short,
                                         w_samples=w_samples)
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
