"""
Entry point: Train the Deep Koopman model.
Implements the full training pipeline and computes the projection matrix D.
"""

import os
import sys
import json
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    N_X, N_U, N_Z, N_W, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    K_PRED, VAL_SPLIT, MODEL_DIR, GAMMA_RIDGE
)
from data.data_loader import load_and_subsample, create_datasets
from model.koopman_network import DeepKoopmanPaper
from model.koopman_trainer import train_model
from model.projection import compute_projection_matrix, save_projection_matrix
from visualization.plot_tsne import plot_tsne_latent_space


def main():
    print("=" * 60)
    print("Deep Koopman Model Training (Paper Algorithm 1)")
    print("=" * 60)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Step 1: Load and subsample data
    print("\n--- Step 1: Loading data ---")
    X_sub, U_sub, norm_params = load_and_subsample()

    # Step 2: Create datasets
    print("\n--- Step 2: Creating datasets ---")
    train_loader, val_loader = create_datasets(
        X_sub, U_sub,
        window_len=K_PRED,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE
    )

    # Step 3: Initialize model
    print("\n--- Step 3: Initializing model ---")
    model = DeepKoopmanPaper(n_x=N_X, n_u=N_U, n_z=N_Z, n_w=N_W)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Architecture:")
    print(f"  Encoder: {N_X} -> 64 -> 128 -> 64 -> {N_Z}")
    print(f"  Decoder: {N_Z} -> 64 -> 32 -> {N_X}")
    print(f"  Linear: A({N_Z}x{N_Z}), B({N_Z}x{N_U}), C({N_Z}x{N_W})")

    # Step 4: Train
    print("\n--- Step 4: Training ---")
    model, training_log = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=device
    )

    # Step 5: Compute projection matrix D
    print("\n--- Step 5: Computing projection matrix D ---")
    model = model.to('cpu')
    model.eval()
    D, r2 = compute_projection_matrix(model, X_sub, gamma=GAMMA_RIDGE)
    save_projection_matrix(D)

    # Step 6: Save Koopman matrices
    print("\n--- Step 6: Saving Koopman matrices ---")
    A, B, C = model.get_matrices()
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, 'koopman_matrices.npz'),
             A=A, B=B, C=C, D=D)
    print(f"  A: {A.shape}, max|eig|={np.max(np.abs(np.linalg.eigvals(A))):.4f}")
    print(f"  B: {B.shape}")
    print(f"  C: {C.shape}")
    print(f"  D: {D.shape}, R^2={r2.mean():.4f}")

    # Save normalization params in output
    with open(os.path.join(MODEL_DIR, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)

    # Step 7: Generate t-SNE visualization (Figure 5)
    print("\n--- Step 7: Generating t-SNE visualization ---")
    plot_tsne_latent_space(model, X_sub)

    # Step 8: Print training summary (Table 6)
    print("\n--- Step 8: Training Summary (Table 6) ---")
    from visualization.plot_tables import print_table_6
    print_table_6(training_log)

    print("\nTraining pipeline complete.")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
