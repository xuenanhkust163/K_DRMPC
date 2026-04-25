# Patch Plan To Match Paper Ground Truth

This document records the concrete patch set to align the repository implementation with the paper source of truth:
- /Users/nevinxue/Desktop/tongji_phd_paper/document_20260425.tex

## Patch 1: Unify State Order With Paper

Target files:
- config.py
- vehicle/bicycle_model.py
- simulation/simulator.py
- controllers/lmpc_controller.py
- controllers/nmpc_controller.py
- controllers/kmpc_controller.py
- controllers/kdrmpc_controller.py
- model/projection.py
- tracks/*.py where state indexing is used

Required change:
- Replace code-wide state order [px, py, v, psi, omega] with [px, py, psi, v, omega].
- Update all index usage:
  - psi index: 2
  - v index: 3
  - omega index: 4
- Update reference extraction in controllers and simulator to use v and omega from indices 3 and 4.
- Update comments and docstrings to avoid stale index hints.

Acceptance:
- A single helper in config.py defines canonical indices and all modules use it.

## Patch 2: Make Vehicle Dynamics Match Paper Equation Form

Target file:
- vehicle/bicycle_model.py

Required change:
- Remove first-order omega dynamics omega_dot = (psi_dot - omega) / tau.
- Implement kinematic consistency:
  - psi_dot = v * tan(delta) / L
  - omega should be algebraically consistent with psi_dot in the discrete update.
- Ensure discrete propagation outputs state in paper order.

Acceptance:
- Discrete step and CasADi dynamics are consistent with the same model equations.

## Patch 3: Enforce Paper-Style Koopman Projection Structure

Target files:
- model/koopman_network.py
- model/projection.py
- controllers/kmpc_controller.py
- controllers/kdrmpc_controller.py

Required change:
- Use fixed selector projections from latent state for D, E, F per paper control-oriented section.
- Do not rely on learned ridge D for core controller constraints and cost.
- Keep optional learned diagnostic projection only for analysis, not control constraints.

Acceptance:
- Controller cost and constraints use paper selectors for position, speed, yaw-rate extraction.

## Patch 4: Train And Use Disturbance Path Consistently

Target files:
- data/data_loader.py
- model/koopman_network.py
- model/koopman_trainer.py

Required change:
- Add disturbance sequence input for training windows where needed by model.
- Use z_next = A z + B u + C w in loss terms where the paper formulation requires disturbance-aware dynamics.
- Ensure C receives identifiable supervision signal.

Acceptance:
- Training log includes disturbance-aware loss components and C is actively used.

## Patch 5: Include Disturbance In K-DRMPC Horizon Dynamics

Target file:
- controllers/kdrmpc_controller.py

Required change:
- Replace nominal-only rollout z_{t+1} = A z_t + B u_t with disturbance-aware trajectory relation used by paper stacked form.
- Use historical disturbance trajectories in the optimization-consistent way.

Acceptance:
- Rollout equations in controller match paper Section 5 compact dynamics assumptions.

## Patch 6: Replace Euclidean Obstacle CVaR Form With Paper Linear Margin Form

Target file:
- controllers/kdrmpc_controller.py

Required change:
- Replace l_nom based on Euclidean distance with paper linear margin:
  - l = H_obs,j * D * z_t - h_obs,j
- Implement dual CVaR constraints with lambda and s_{t,i} according to paper equation set.
- Remove ad hoc Cw_norm and w_norm surrogates unless mathematically derived from chosen norm and dual expression.

Acceptance:
- CVaR constraints are equation-consistent with paper Section 5 derivation.

## Patch 7: Remove Sample Truncation To 20 For Main Results

Target file:
- controllers/kdrmpc_controller.py

Required change:
- Remove hard truncation MAX_OPT_SAMPLES = 20 in default paper-reproduction mode.
- Use N_DISTURBANCE_SAMPLES from config (default 100).
- If runtime mode is needed, gate down-sampling behind an explicit fast mode flag.

Acceptance:
- Default experiments use N = 100 samples as paper setup.

## Patch 8: Enforce Full-Horizon Constraint Coverage In Paper Mode

Target files:
- controllers/kmpc_controller.py
- controllers/kdrmpc_controller.py

Required change:
- Remove sparse check_steps in default paper mode.
- Evaluate obstacle and safety constraints across all horizon steps.
- Remove obstacle proximity filtering in paper mode.

Acceptance:
- Constraints are enforced for all t in horizon for all configured obstacles.

## Patch 9: Align Disturbance Generator With Paper Distribution

Target file:
- disturbance/disturbance_generator.py

Required change:
- Set mixture to two zero-mean Gaussian components.
- Use covariance levels 0.01 I and 0.05 I in default paper mode.
- Keep existing 3-component generator only as optional non-paper stress mode.

Acceptance:
- Default generated disturbance statistics match paper setup text.

## Patch 10: Enforce Missing State Constraints In Controllers

Target files:
- controllers/lmpc_controller.py
- controllers/nmpc_controller.py
- controllers/kmpc_controller.py
- controllers/kdrmpc_controller.py

Required change:
- Add yaw-rate bound constraints where required by paper.
- Ensure speed constraints are present in K-MPC and K-DRMPC, not only LMPC and NMPC.
- Ensure steering-rate and input bounds remain active.

Acceptance:
- Constraint set matches paper nominal and Koopman formulations.

## Patch 11: Match Reported Experiment Step Count In Default Reproduction

Target file:
- config.py

Required change:
- Set default simulation step count to 2000 for paper reproduction profile.
- Keep longer runs available under optional extended mode.

Acceptance:
- Default run scripts produce paper-comparable simulation duration.

## Patch 12: Add Reproduction Profiles

Target files:
- config.py
- run_simulation.py
- README.md

Required change:
- Add explicit mode switch:
  - paper_strict: fully paper-faithful defaults
  - fast_debug: current speed-oriented approximations
- Document differences clearly.

Acceptance:
- Running simulation in paper_strict mode reproduces paper assumptions without hidden shortcuts.

## Execution Order

1. Patch 1 and Patch 2
2. Patch 10
3. Patch 9
4. Patch 4 and Patch 5
5. Patch 6 and Patch 7
6. Patch 8
7. Patch 11 and Patch 12
8. Re-run training and simulation, then regenerate analysis outputs

## Validation Checklist

- Unit-level checks:
  - State index sanity checks across all controllers
  - Dynamics consistency between numpy and CasADi versions
- Training checks:
  - Disturbance-aware losses decrease
  - C matrix has non-trivial learned effect
- Control checks:
  - K-DRMPC constraints satisfy paper equations
  - Full-horizon constraints active in logs
- Experiment checks:
  - Default config matches paper parameters
  - Results regenerate without fast-mode approximations
