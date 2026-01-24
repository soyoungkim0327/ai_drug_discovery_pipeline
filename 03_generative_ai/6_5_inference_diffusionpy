# 6-5.Inference_Diffusion.py (FIXED)
# Description: Diffusion-based Generative Model Inference & Trajectory Visualization
# Function: Generates new data points from Gaussian Noise via Reverse Denoising Process.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random  # Added for reproducibility

# --- [Reproducibility Configuration] ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False
# --------------------

# -----------------------------
# 1. Model Architecture Definition
#    (Aligned with Training Architecture in '5.Simple_Diffusion.py')
# -----------------------------
class SimpleDiffusion(nn.Module):
    def __init__(self, n_steps=50):
        super().__init__()
        self.n_steps = n_steps
        # MLP-based Denoising Network
        self.net = nn.Sequential(
            nn.Linear(2 + 1, 64),  # Input: Coordinate(2) + Time Embedding(1)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)       # Output: Predicted Noise(2)
        )

    def forward(self, x, t):
        """
        Forward Pass for Noise Prediction
        t: Time step tensor (Shape: [B,])
        - Normalized to [0, 1] range for embedding
        """
        t = t.view(-1, 1).float() / float(self.n_steps)
        x_input = torch.cat([x, t], dim=1)
        return self.net(x_input)


# -----------------------------
# 2. Environment & Path Configuration
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_diffusion.pth")

n_steps = 50


def safe_torch_load(path):
    # Secure model loading (Support 'weights_only' for newer PyTorch versions)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# -----------------------------
# 3. Auto-Training Protocol (Fallback Mechanism)
# -----------------------------
def train_and_save(model_path, n_steps=50, epochs=200, lr=0.01):
    print("[System] Pre-trained weights not found. Initiating auto-training sequence...")

    model = SimpleDiffusion(n_steps=n_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Synthetic Target Data: Distribution centered at [1.0, 1.0]
    real_data = torch.tensor([[1.0, 1.0]] * 256, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Random Time-step Sampling
        t = torch.randint(0, n_steps, (real_data.size(0),), device=device)

        # Gaussian Noise Injection
        noise = torch.randn_like(real_data)

        # Forward Diffusion Process: q(x_t | x_0)
        alpha = 1 - (t.float() / n_steps).view(-1, 1)
        noisy_data = real_data * alpha + noise * (1 - alpha)

        # Noise Prediction
        predicted_noise = model(noisy_data, t)

        # Objective Function: MSE Loss
        loss = nn.MSELoss()(predicted_noise, noise)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    # Save Trained Weights
    torch.save(model.state_dict(), model_path)
    print(f"Model Checkpoint Saved: {model_path}")

    return model


# -----------------------------
# 4. Model Loading Strategy
# -----------------------------
def load_or_train():
    model = SimpleDiffusion(n_steps=n_steps).to(device)

    if os.path.exists(model_path):
        try:
            state = safe_torch_load(model_path)
            model.load_state_dict(state)
            model.eval()
            print(f"Diffusion Model Loaded Successfully: {model_path}")
            return model
        except Exception as e:
            print("[Error] Failed to load existing checkpoint:", type(e).__name__, e)
            print("   -> Initiating Retraining Protocol.")
            return train_and_save(model_path, n_steps=n_steps)

    # Train if file does not exist
    return train_and_save(model_path, n_steps=n_steps)


# -----------------------------
# 5. Generative Process (Reverse Denoising)
# -----------------------------
@torch.no_grad()
def generate_points(model, num_points=500):
    model.eval()

    # Initialization: Sampling from Isotropic Gaussian Noise
    x = torch.randn(num_points, 2, device=device)

    # Reverse Diffusion Process: p(x_{t-1} | x_t)
    for t in reversed(range(n_steps)):
        t_tensor = torch.full((num_points,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_tensor)
        # Denoising Step (Simplified Update Rule)
        x = x - predicted_noise * 0.1

    return x.detach().cpu().numpy()


@torch.no_grad()
def generate_trajectory(model):
    """
    Trace the reconstruction trajectory for a single data point.
    Used for visualizing the denoising dynamics.
    """
    model.eval()
    x = torch.randn(1, 2, device=device)
    traj = [x.detach().cpu().numpy()]

    for t in reversed(range(n_steps)):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        predicted_noise = model(x, t_tensor)
        x = x - predicted_noise * 0.1
        traj.append(x.detach().cpu().numpy())

    return traj, x.detach().cpu().numpy()


def save_scatter(points_np, out_png="diffusion_generated.png"):
    plt.figure(figsize=(6, 6))
    plt.scatter(points_np[:, 0], points_np[:, 1], s=6, alpha=0.5)
    plt.title("Generated Distribution by Simple Diffusion")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(current_dir, out_png), dpi=150)
    print(f"Distribution Plot Saved: {out_png}")


def save_trajectory(traj, out_png="diffusion_process.png"):
    traj_np = np.array(traj)  # Shape: (steps, 1, 2)

    plt.figure(figsize=(10, 6))

    # Ground Truth / Target Marker
    plt.scatter([1.0], [1.0], s=250, marker="*", label="Target (1.0, 1.0)")

    steps = traj_np.shape[0]
    for i in range(steps):
        pos = traj_np[i, 0]
        # Gradient Visualization (Alpha increases as it approaches t=0)
        plt.scatter(pos[0], pos[1], s=30, alpha=0.25 + 0.75 * (i / steps))
        if i > 0:
            prev = traj_np[i - 1, 0]
            plt.plot([prev[0], pos[0]], [prev[1], pos[1]], alpha=0.25)

    plt.title("Denoising Trajectory: Gaussian Noise -> Target Structure")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.savefig(os.path.join(current_dir, out_png), dpi=150)
    print(f"Trajectory Plot Saved: {out_png}")


# -----------------------------
# 6. Execution Pipeline
# -----------------------------
if __name__ == "__main__":
    model = load_or_train()

    print("\n[Inference] Generating new structures via Reverse Diffusion...")
    pts = generate_points(model, num_points=500)
    save_scatter(pts, out_png="diffusion_generated.png")

    traj, final_xy = generate_trajectory(model)
    save_trajectory(traj, out_png="diffusion_process.png")

    print(f"\n[Validation] Single Sample Final Coordinate: {final_xy} (Expected vicinity: [1.0, 1.0])")
    print("Process Completed.")