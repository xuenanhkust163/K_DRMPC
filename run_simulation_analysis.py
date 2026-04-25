"""
Integrated entry point: run simulation then analysis.

Before each run, clear files under:
- _output/results
- _output/figures
"""

import argparse
import os
import subprocess
import sys

from config import FIGURES_DIR, RESULTS_DIR


def clear_output_files(dir_path):
    """Delete all files under dir_path recursively, preserving root directory."""
    os.makedirs(dir_path, exist_ok=True)

    for root, dirs, files in os.walk(dir_path, topdown=False):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
            except OSError as exc:
                print(f"Warning: failed to delete file {file_path}: {exc}")

        for dirname in dirs:
            subdir_path = os.path.join(root, dirname)
            try:
                # Remove empty subdirs after file cleanup.
                os.rmdir(subdir_path)
            except OSError:
                # Ignore non-empty or protected directories.
                pass


def run_command(cmd):
    """Run a command and stream output."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simulation + analysis with pre-run output cleanup."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass through to run_simulation.py for step-level logs.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Pass through to run_simulation.py for 300-step quick validation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Pass through to run_simulation.py as max simulation steps (default: 200).",
    )
    parser.add_argument(
        "--cost-profile",
        type=str,
        choices=["default", "tracking-first", "progress-first"],
        default="default",
        help="Pass through to run_simulation.py cost profile.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Simulation + Analysis Pipeline")
    print("=" * 60)

    print("\nCleaning previous outputs...")
    clear_output_files(RESULTS_DIR)
    clear_output_files(FIGURES_DIR)
    print(f"  Cleared files in: {RESULTS_DIR}")
    print(f"  Cleared files in: {FIGURES_DIR}")

    sim_cmd = [sys.executable, "run_simulation.py"]
    if args.fast and args.steps == 200:
        sim_cmd.append("--fast")
    else:
        sim_cmd.extend(["--steps", str(args.steps)])
    if args.verbose:
        sim_cmd.append("--verbose")
    sim_cmd.extend(["--cost-profile", args.cost_profile])

    run_command(sim_cmd)
    run_command([sys.executable, "run_analysis.py"])

    print("\n" + "=" * 60)
    print("Simulation + analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
