import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
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

# 1. Environment Configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 2. Diffusion Model Architecture Definition
class SimpleDiffusion(nn.Module):
    def __init__(self, n_steps=50):
        super().__init__()
        self.n_steps = n_steps
        # MLP based Denoising Network
        self.net = nn.Sequential(
            nn.Linear(2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t):
        # Time embedding and concatenation
        t_input = t.view(-1, 1).float() / self.n_steps
        x_input = torch.cat([x, t_input], dim=1)
        return self.net(x_input)

# 3. Training Hyperparameters & Initialization
model = SimpleDiffusion()
optimizer = optim.Adam(model.parameters(), lr=0.01)
n_steps = 50
real_data = torch.tensor([[1.0, 1.0]] * 100) # Target Distribution (Ground Truth)

# ============================================================
# [Main Loop] Training with Early Stopping Strategy
# ============================================================
print("Start Diffusion Model Training (Early Stopping applied)...")

patience = 20           # Patience threshold
patience_counter = 0    
best_loss = float('inf')
max_epochs = 1000       # Max epochs

for epoch in range(1, max_epochs + 1):
    optimizer.zero_grad()
    
    # (1) Forward Process: Noise Injection
    t = torch.randint(0, n_steps, (100,))
    noise = torch.randn_like(real_data)
    alpha = 1 - (t.float() / n_steps).view(-1, 1)
    noisy_data = real_data * alpha + noise * (1 - alpha)
    
    # (2) Reverse Process: Noise Prediction & Loss Calculation
    predicted_noise = model(noisy_data, t)
    loss = nn.MSELoss()(predicted_noise, noise)
    
    loss.backward()
    optimizer.step()
    
    current_loss = loss.item()

    # --- Early Stopping Logic ---
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
        # Save Best Model Weights
        torch.save(model.state_dict(), 'best_diffusion.pth')
        if epoch % 20 == 0: # Log every 20 epochs
            print(f"Epoch {epoch:03d}, Loss: {current_loss:.6f} [New Best Model Saved]")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nStop Training (Early Stopping Triggered)")
            print(f"   - Stopped at epoch {epoch}. (Min Loss: {best_loss:.6f})")
            break

print("Training Completed. Proceeding to Inference.")

# 4. Inference & Visualization (Load Best Model)
# Load trained state dictionary for inference
if os.path.exists('best_diffusion.pth'):
    model.load_state_dict(torch.load('best_diffusion.pth'))
    print("Best model checkpoint loaded: 'best_diffusion.pth'")

x = torch.randn(1, 2) 
trajectory = [x.detach().numpy()]

with torch.no_grad():
    for t in reversed(range(n_steps)):
        t_tensor = torch.tensor([t])
        predicted_noise = model(x, t_tensor)
        alpha = 1 - (t / n_steps)
        x = x - predicted_noise * 0.1
        trajectory.append(x.detach().numpy())

print(f"Generated Coordinates: {x.numpy()}")

# 5. Visualization of Diffusion Trajectory
def visualize_diffusion():
    print("\nVisualizing Diffusion Trajectory...")
    traj_np = np.array(trajectory)
    plt.figure(figsize=(10, 6))
    plt.scatter([1.0], [1.0], c='red', s=300, marker='*', label='Goal')
    
    steps = traj_np.shape[0]
    for i in range(steps):
        alpha = i / steps
        color = (0, 0, 1, alpha)
        pos = traj_np[i, 0]
        plt.scatter(pos[0], pos[1], color=color, s=50)
        if i > 0:
            prev_pos = traj_np[i-1, 0]
            plt.plot([prev_pos[0], pos[0]], [prev_pos[1], pos[1]], color='gray', alpha=0.3)

    plt.title("Diffusion Process with Early Stopping")
    plt.grid(True, linestyle='--')
    plt.savefig('diffusion_early_stopping.png')
    print("Visualization saved: 'diffusion_early_stopping.png'")

visualize_diffusion()