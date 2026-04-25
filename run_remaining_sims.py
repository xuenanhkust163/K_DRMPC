"""
Run remaining simulations: K-MPC and K-DRMPC on Lusail, all 4 methods on Custom track,
plus robustness and sensitivity analyses.
"""
import os
import sys
import json
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MODEL_DIR, RESULTS_DIR, N_DISTURBANCE_SAMPLES,
    SIGMA_VALUES, THETA_VALUES, EPSILON_VALUES,
    THETA_WASSERSTEIN, EPSILON_CVAR, MAX_SIM_STEPS
)
from model.koopman_trainer import load_trained_model
from model.projection import load_projection_matrix
from tracks.lusail_track import LusailTrack
from tracks.custom_track import CustomWindingTrack
from controllers.lmpc_controller import LMPCController
from controllers.nmpc_controller import NMPCController
from controllers.kmpc_controller import KMPCController
from controllers.kdrmpc_controller import KDRMPCController
from disturbance.disturbance_generator import DisturbanceGenerator
from simulation.simulator import Simulator


def load_koopman():
    model = load_trained_model()
    D = load_projection_matrix()
    with open(os.path.join(MODEL_DIR, 'norm_params.json')) as f:
        norm_params = json.load(f)
    return model, D, norm_params


def run_and_save(method_name, track, controller, dist_gen, max_steps):
    """Run simulation and save result."""
    simulator = Simulator(track, controller, dist_gen)
    result = simulator.run(max_steps=max_steps, verbose=True)
    Simulator.save_result(result, f"{method_name}_{track.__class__.__name__}.pkl")
    return result


def main():
    total_start = time.time()
    model, D, norm_params = load_koopman()
    dist_gen = DisturbanceGenerator(sigma=0.05)
    w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

    # ===== Lusail Circuit: K-MPC and K-DRMPC =====
    print("\n" + "=" * 60)
    print("LUSAIL: K-MPC")
    print("=" * 60)
    lusail = LusailTrack()

    kmpc = KMPCController(model, D, norm_params)
    run_and_save('K-MPC', lusail, kmpc, dist_gen, MAX_SIM_STEPS)

    print("\n" + "=" * 60)
    print("LUSAIL: K-DRMPC")
    print("=" * 60)
    kdrmpc = KDRMPCController(model, D, norm_params, disturbance_samples=w_empirical)
    run_and_save('K-DRMPC', lusail, kdrmpc, dist_gen, MAX_SIM_STEPS)

    # ===== Custom Winding Track: All 4 methods =====
    print("\n" + "=" * 60)
    print("CUSTOM: LMPC")
    print("=" * 60)
    custom = CustomWindingTrack()

    lmpc = LMPCController()
    run_and_save('LMPC', custom, lmpc, dist_gen, MAX_SIM_STEPS)

    print("\n" + "=" * 60)
    print("CUSTOM: NMPC")
    print("=" * 60)
    nmpc = NMPCController()
    run_and_save('NMPC', custom, nmpc, dist_gen, MAX_SIM_STEPS)

    print("\n" + "=" * 60)
    print("CUSTOM: K-MPC")
    print("=" * 60)
    kmpc2 = KMPCController(model, D, norm_params)
    run_and_save('K-MPC', custom, kmpc2, dist_gen, MAX_SIM_STEPS)

    print("\n" + "=" * 60)
    print("CUSTOM: K-DRMPC")
    print("=" * 60)
    kdrmpc2 = KDRMPCController(model, D, norm_params, disturbance_samples=w_empirical)
    run_and_save('K-DRMPC', custom, kdrmpc2, dist_gen, MAX_SIM_STEPS)

    # ===== Robustness Analysis (Table 11) =====
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS (Table 11)")
    print("=" * 60)
    for sigma in SIGMA_VALUES:
        print(f"\n--- sigma={sigma} ---")
        dg = DisturbanceGenerator(sigma=sigma)
        w_emp = dg.get_empirical_samples(N_DISTURBANCE_SAMPLES)
        ctrl = KDRMPCController(model, D, norm_params, disturbance_samples=w_emp)
        sim = Simulator(lusail, ctrl, dg)
        result = sim.run(max_steps=500, verbose=False)
        Simulator.save_result(result, f"robustness_sigma_{sigma}.pkl")
        print(f"  Done: {result.total_steps} steps, lap={result.lap_completed}")

    # ===== Sensitivity: theta (Table 14) =====
    print("\n" + "=" * 60)
    print("SENSITIVITY: THETA (Table 14)")
    print("=" * 60)
    for theta in THETA_VALUES:
        print(f"\n--- theta={theta} ---")
        ctrl = KDRMPCController(model, D, norm_params,
                                disturbance_samples=w_empirical, theta=theta)
        sim = Simulator(lusail, ctrl, dist_gen)
        result = sim.run(max_steps=500, verbose=False)
        Simulator.save_result(result, f"sensitivity_theta_{theta}.pkl")
        print(f"  Done: {result.total_steps} steps, lap={result.lap_completed}")

    # ===== Sensitivity: epsilon (Table 15) =====
    print("\n" + "=" * 60)
    print("SENSITIVITY: EPSILON (Table 15)")
    print("=" * 60)
    for epsilon in EPSILON_VALUES:
        print(f"\n--- epsilon={epsilon} ---")
        ctrl = KDRMPCController(model, D, norm_params,
                                disturbance_samples=w_empirical, epsilon=epsilon)
        sim = Simulator(lusail, ctrl, dist_gen)
        result = sim.run(max_steps=500, verbose=False)
        Simulator.save_result(result, f"sensitivity_epsilon_{epsilon}.pkl")
        print(f"  Done: {result.total_steps} steps, lap={result.lap_completed}")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All simulations complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
