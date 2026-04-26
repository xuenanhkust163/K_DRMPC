"""
Entry point: Evaluate a trained Deep Koopman checkpoint on the validation split.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BATCH_SIZE, FIGURES_DIR, GAMMA_RIDGE, K_PRED, MODEL_DIR, VAL_SPLIT
from data.data_loader import create_datasets, load_and_subsample
from model.koopman_network import koopman_loss
from model.koopman_trainer import load_trained_model
from model.projection import compute_projection_matrix
from visualization.plot_model_evaluation import plot_rmse_comparison


STATE_NAMES = ["px_norm", "py_norm", "psi", "v", "omega"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Koopman checkpoint on the validation split."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(MODEL_DIR, "best_koopman_model.pth"),
        help="Checkpoint path. Defaults to _output/models/best_koopman_model.pth.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(MODEL_DIR, "koopman_model_evaluation.json"),
        help="Where to save the evaluation summary JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=FIGURES_DIR,
        help="Directory where evaluation figures are saved.",
    )
    return parser.parse_args()


def mean_stack(values):
    return np.mean(np.stack(values), axis=0)


def evaluate_model(model, val_loader):
    losses = {"total": 0.0, "recon": 0.0, "linear": 0.0, "pred": 0.0}
    one_step_sq = []
    multi_step_sq = []
    baseline_one_step_sq = []
    baseline_multi_step_sq = []
    num_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_x, batch_u in val_loader:
            _, loss_dict = koopman_loss(model, batch_x, batch_u)
            for key in losses:
                losses[key] += float(loss_dict[key])

            x0 = batch_x[:, 0, :]
            x1 = batch_x[:, 1, :]

            one_step_pred = model(x0, batch_u[:, 0, :])["x_next_pred"]
            multi_step_pred, _ = model.multi_step_predict(x0, batch_u)
            multi_step_target = batch_x[:, 1:, :]

            baseline_one_step = x0
            baseline_multi_step = x0[:, None, :].expand_as(multi_step_target)

            one_step_sq.append(torch.mean((one_step_pred - x1) ** 2, dim=0).cpu().numpy())
            multi_step_sq.append(
                torch.mean((multi_step_pred - multi_step_target) ** 2, dim=(0, 1)).cpu().numpy()
            )
            baseline_one_step_sq.append(
                torch.mean((baseline_one_step - x1) ** 2, dim=0).cpu().numpy()
            )
            baseline_multi_step_sq.append(
                torch.mean((baseline_multi_step - multi_step_target) ** 2, dim=(0, 1)).cpu().numpy()
            )
            num_batches += 1

    for key in losses:
        losses[key] /= max(num_batches, 1)

    return {
        "losses": losses,
        "one_step_rmse": np.sqrt(mean_stack(one_step_sq)),
        "multi_step_rmse": np.sqrt(mean_stack(multi_step_sq)),
        "baseline_one_step_rmse": np.sqrt(mean_stack(baseline_one_step_sq)),
        "baseline_multi_step_rmse": np.sqrt(mean_stack(baseline_multi_step_sq)),
        "num_batches": num_batches,
    }


def zip_state_metrics(values):
    return {name: float(value) for name, value in zip(STATE_NAMES, values)}


def build_summary(args, checkpoint, training_log, eval_result, r2, norm_params, x_sub, u_sub):
    best_epoch = int(np.argmin(training_log["val_loss"]) + 1)

    summary = {
        "model_path": os.path.abspath(args.model_path),
        "figure_dir": os.path.abspath(args.figure_dir),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "training_log_best_epoch": best_epoch,
        "training_log_best_val_loss": float(np.min(training_log["val_loss"])),
        "training_log_final_val_loss": float(training_log["val_loss"][-1]),
        "evaluated_val_losses": eval_result["losses"],
        "one_step_rmse": zip_state_metrics(eval_result["one_step_rmse"]),
        "multi_step_rmse": zip_state_metrics(eval_result["multi_step_rmse"]),
        "baseline_one_step_rmse": zip_state_metrics(eval_result["baseline_one_step_rmse"]),
        "baseline_multi_step_rmse": zip_state_metrics(eval_result["baseline_multi_step_rmse"]),
        "projection_r2": {
            "v": float(r2[0]),
            "omega": float(r2[1]),
            "mean": float(np.mean(r2)),
        },
        "position_rmse_meters": {
            "one_step_px": float(eval_result["one_step_rmse"][0] * norm_params["px_std"]),
            "one_step_py": float(eval_result["one_step_rmse"][1] * norm_params["py_std"]),
            "multi_step_px": float(eval_result["multi_step_rmse"][0] * norm_params["px_std"]),
            "multi_step_py": float(eval_result["multi_step_rmse"][1] * norm_params["py_std"]),
        },
        "dataset_sizes": {
            "states": int(x_sub.shape[0]),
            "controls": int(u_sub.shape[0]),
            "val_batches": eval_result["num_batches"],
        },
    }
    return summary


def print_summary(summary):
    print("\n" + "=" * 60)
    print("Koopman Model Evaluation")
    print("=" * 60)
    print(f"Model: {summary['model_path']}")
    print(
        f"Checkpoint epoch: {summary['checkpoint_epoch']} | "
        f"checkpoint val_loss: {summary['checkpoint_val_loss']:.6f}"
    )
    print(
        f"Best epoch in log: {summary['training_log_best_epoch']} | "
        f"best val_loss: {summary['training_log_best_val_loss']:.6f} | "
        f"final val_loss: {summary['training_log_final_val_loss']:.6f}"
    )

    print("\n[Validation Losses]")
    for key, value in summary["evaluated_val_losses"].items():
        print(f"  {key:<8} {value:.6f}")

    print("\n[One-step RMSE]")
    for key in STATE_NAMES:
        model_val = summary["one_step_rmse"][key]
        base_val = summary["baseline_one_step_rmse"][key]
        print(f"  {key:<8} model={model_val:10.6f} | baseline={base_val:10.6f}")

    print("\n[Multi-step RMSE]")
    for key in STATE_NAMES:
        model_val = summary["multi_step_rmse"][key]
        base_val = summary["baseline_multi_step_rmse"][key]
        print(f"  {key:<8} model={model_val:10.6f} | baseline={base_val:10.6f}")

    print("\n[Projection R^2]")
    print(
        f"  v={summary['projection_r2']['v']:.6f} | "
        f"omega={summary['projection_r2']['omega']:.6f} | "
        f"mean={summary['projection_r2']['mean']:.6f}"
    )

    print("\n[Position RMSE in meters]")
    for key, value in summary["position_rmse_meters"].items():
        print(f"  {key:<12} {value:.6f}")

    print("=" * 60)


def main():
    args = parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)

    log_path = os.path.join(MODEL_DIR, "training_log.json")
    with open(log_path, "r") as f:
        training_log = json.load(f)

    x_sub, u_sub, norm_params = load_and_subsample()
    _, val_loader = create_datasets(
        x_sub,
        u_sub,
        window_len=K_PRED,
        val_split=VAL_SPLIT,
        batch_size=args.batch_size,
    )

    model = load_trained_model(args.model_path)
    eval_result = evaluate_model(model, val_loader)
    _, r2 = compute_projection_matrix(model, x_sub, gamma=GAMMA_RIDGE)

    summary = build_summary(
        args,
        checkpoint,
        training_log,
        eval_result,
        r2,
        norm_params,
        x_sub,
        u_sub,
    )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    figure_path = plot_rmse_comparison(summary, save_dir=args.figure_dir)
    summary["rmse_figure_path"] = os.path.abspath(figure_path)

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"Saved evaluation JSON to: {args.output_json}")
    print(f"Saved evaluation figure to: {figure_path}")


if __name__ == "__main__":
    main()