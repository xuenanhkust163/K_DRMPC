"""Reusable tracking-cost builders for Koopman MPC controllers."""

from dataclasses import dataclass
from typing import Optional, Union

import casadi as ca


@dataclass
class MinSpeedRule:
    """Soft minimum-speed rule used by DR formulations."""

    floor_abs: float
    floor_ratio: float

    def floor(self, v_ref: float) -> float:
        return max(self.floor_abs, self.floor_ratio * float(v_ref))


class DefaultTrackingCostBuilder:
    """Default stage-cost builder for Koopman trajectory tracking."""

    def stage_cost(
        self,
        *,
        opti,
        t: int,
        z_t,
        u_t,
        u_prev,
        u_prev_step,
        y_t,
        y_ref_t,
        ref_psi_t: float,
        ref_px_norm_t: float,
        ref_py_norm_t: float,
        d_pos_ca,
        d_psi_ca,
        q,
        r,
        q_psi: float,
        q_progress: float,
        q_pos: float,
        add_position_term: bool,
        add_abs_u_term: bool = False,
        r_abs=None,
        min_speed_rule: Optional[MinSpeedRule] = None,
        v_slack_t=None,
    ):
        """Return stage cost term and add optional soft constraints to opti."""
        stage = 0

        # [v, omega] tracking
        y_err = y_t - y_ref_t
        stage += ca.mtimes([y_err.T, q, y_err])

        # Heading tracking with wrapped angle error
        psi_t = ca.mtimes(d_psi_ca, z_t)[0]
        psi_err = ca.atan2(ca.sin(psi_t - float(ref_psi_t)),
                           ca.cos(psi_t - float(ref_psi_t)))
        stage += q_psi * (psi_err ** 2)

        # Forward-progress tracking
        v_forward = y_t[0] * ca.cos(psi_err)
        stage += q_progress * ((v_forward - float(y_ref_t[0])) ** 2)

        # Optional soft minimum-speed constraint
        if min_speed_rule is not None and v_slack_t is not None:
            v_floor = min_speed_rule.floor(float(y_ref_t[0]))
            opti.subject_to(y_t[0] + v_slack_t >= v_floor)

        # Position term (typically every 4th step)
        if add_position_term:
            pos = ca.mtimes(d_pos_ca, z_t)
            pos_err_x = pos[0] - float(ref_px_norm_t)
            pos_err_y = pos[1] - float(ref_py_norm_t)
            stage += q_pos * (pos_err_x ** 2 + pos_err_y ** 2)

        # Input increment penalty
        if t == 0:
            du = u_t - u_prev
        else:
            du = u_t - u_prev_step
        stage += ca.mtimes([du.T, r, du])

        # Optional absolute input penalty
        if add_abs_u_term and r_abs is not None:
            stage += ca.mtimes([u_t.T, r_abs, u_t])

        return stage


class WeightedTrackingCostBuilder(DefaultTrackingCostBuilder):
    """Default builder with simple multiplicative profile scales."""

    def __init__(
        self,
        *,
        q_psi_scale: float = 1.0,
        q_progress_scale: float = 1.0,
        q_pos_scale: float = 1.0,
        du_scale: float = 1.0,
        abs_u_scale: float = 1.0,
    ):
        self.q_psi_scale = float(q_psi_scale)
        self.q_progress_scale = float(q_progress_scale)
        self.q_pos_scale = float(q_pos_scale)
        self.du_scale = float(du_scale)
        self.abs_u_scale = float(abs_u_scale)

    def stage_cost(self, **ctx):
        stage = super().stage_cost(**ctx)

        # Rebuild key terms with profile multipliers by applying incremental deltas.
        # This keeps API compatibility while allowing fast profile switching.
        y_t = ctx["y_t"]
        y_ref_t = ctx["y_ref_t"]
        z_t = ctx["z_t"]
        d_psi_ca = ctx["d_psi_ca"]
        d_pos_ca = ctx["d_pos_ca"]

        q_psi = float(ctx["q_psi"])
        q_progress = float(ctx["q_progress"])
        q_pos = float(ctx["q_pos"])
        ref_psi_t = float(ctx["ref_psi_t"])
        ref_px_norm_t = float(ctx["ref_px_norm_t"])
        ref_py_norm_t = float(ctx["ref_py_norm_t"])

        t = int(ctx["t"])
        u_t = ctx["u_t"]
        u_prev = ctx["u_prev"]
        u_prev_step = ctx["u_prev_step"]
        r = ctx["r"]

        psi_t = ca.mtimes(d_psi_ca, z_t)[0]
        psi_err = ca.atan2(ca.sin(psi_t - ref_psi_t), ca.cos(psi_t - ref_psi_t))
        v_forward = y_t[0] * ca.cos(psi_err)

        base_psi = q_psi * (psi_err ** 2)
        base_progress = q_progress * ((v_forward - float(y_ref_t[0])) ** 2)

        stage += (self.q_psi_scale - 1.0) * base_psi
        stage += (self.q_progress_scale - 1.0) * base_progress

        if bool(ctx["add_position_term"]):
            pos = ca.mtimes(d_pos_ca, z_t)
            pos_err_x = pos[0] - ref_px_norm_t
            pos_err_y = pos[1] - ref_py_norm_t
            base_pos = q_pos * (pos_err_x ** 2 + pos_err_y ** 2)
            stage += (self.q_pos_scale - 1.0) * base_pos

        if t == 0:
            du = u_t - u_prev
        else:
            du = u_t - u_prev_step
        base_du = ca.mtimes([du.T, r, du])
        stage += (self.du_scale - 1.0) * base_du

        if bool(ctx["add_abs_u_term"]) and (ctx.get("r_abs") is not None):
            base_abs_u = ca.mtimes([u_t.T, ctx["r_abs"], u_t])
            stage += (self.abs_u_scale - 1.0) * base_abs_u

        return stage


TRACKING_COST_PROFILES = {
    "default": DefaultTrackingCostBuilder,
    # 更强调中心线/姿态/平滑，抑制过度激进推进。
    "tracking-first": lambda: WeightedTrackingCostBuilder(
        q_psi_scale=1.6,
        q_progress_scale=0.75,
        q_pos_scale=1.4,
        du_scale=1.2,
        abs_u_scale=1.2,
    ),
    # 更强调前进效率，适度放松位置/平滑保守性。
    "progress-first": lambda: WeightedTrackingCostBuilder(
        q_psi_scale=0.9,
        q_progress_scale=1.7,
        q_pos_scale=0.8,
        du_scale=0.9,
        abs_u_scale=0.9,
    ),
}


def resolve_tracking_cost_builder(
    builder: Optional[Union[str, DefaultTrackingCostBuilder]] = None,
    profile: str = "default",
):
    """Resolve a builder name/object to a cost builder instance."""
    if builder is None:
        if profile not in TRACKING_COST_PROFILES:
            raise ValueError(
                f"Unknown cost profile: {profile}. "
                f"Available: {', '.join(TRACKING_COST_PROFILES.keys())}"
            )
        factory = TRACKING_COST_PROFILES[profile]
        return factory() if callable(factory) else factory
    if isinstance(builder, str):
        if builder in TRACKING_COST_PROFILES:
            factory = TRACKING_COST_PROFILES[builder]
            return factory() if callable(factory) else factory
        raise ValueError(f"Unknown tracking cost builder: {builder}")
    if hasattr(builder, "stage_cost"):
        return builder
    raise TypeError(
        "cost_builder must be None, a known profile name, "
        "or an object with stage_cost(...)"
    )
