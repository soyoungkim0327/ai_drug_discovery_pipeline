"""5_diffusion_simple_es_p.py

A toy diffusion model (2D point) with early stopping, plus trajectory visualization.

Changes (professional hygiene):
- main() entry-point (prevents side effects on import)
- Save checkpoint/plots next to this script
- Writes small metadata JSON

Run:
    python 03_generative_ai/5_diffusion_simple_es_p.py
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SimpleDiffusion(nn.Module):
    def __init__(self, n_steps: int = 50):
        super().__init__()
        self.n_steps = n_steps
        self.net = nn.Sequential(
            nn.Linear(2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x, t):
        t_input = t.view(-1, 1).float() / float(self.n_steps)
        x_input = torch.cat([x, t_input], dim=1)
        return self.net(x_input)


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def visualize_trajectory(trajectory, out_path: Path):
    traj_np = np.array(trajectory)

    plt.figure(figsize=(10, 6))
    plt.scatter([1.0], [1.0], s=300, marker='*', label='Target (1.0, 1.0)')

    steps = traj_np.shape[0]
    for i in range(steps):
        alpha = i / steps
        # Explicit color is ok here because this is a demo plot.
        color = (0, 0, 1, alpha)
        pos = traj_np[i, 0]
        plt.scatter(pos[0], pos[1], color=color, s=50)
        if i > 0:
            prev_pos = traj_np[i - 1, 0]
            plt.plot([prev_pos[0], pos[0]], [prev_pos[1], pos[1]], color='gray', alpha=0.3)

    plt.title('Diffusion Process with Early Stopping')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    seed = 42
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    script_dir = Path(__file__).resolve().parent
    ckpt_path = script_dir / 'best_diffusion.pth'
    meta_path = ckpt_path.with_suffix('.meta.json')
    plot_path = script_dir / 'diffusion_early_stopping.png'

    n_steps = 50
    batch_size = 100

    model = SimpleDiffusion(n_steps=n_steps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Target data: distribution centered at (1,1)
    real_data = torch.tensor([[1.0, 1.0]] * batch_size, device=device)

    print('Start Diffusion Model Training (Early Stopping applied)...')

    patience = 20
    patience_counter = 0
    best_loss = float('inf')
    max_epochs = 1000

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        t = torch.randint(0, n_steps, (batch_size,), device=device)
        noise = torch.randn_like(real_data)

        alpha = 1 - (t.float() / n_steps).view(-1, 1)
        noisy_data = real_data * alpha + noise * (1 - alpha)

        predicted_noise = model(noisy_data, t)
        loss = nn.MSELoss()(predicted_noise, noise)

        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())

        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            if epoch % 20 == 0 or epoch == 1:
                print(f'Epoch {epoch:03d} | loss={current_loss:.6f} [best -> saved {ckpt_path.name}]')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('\nEarly stopping triggered.')
                print(f'   - Stopped at epoch {epoch}. (Min loss: {best_loss:.6f})')
                break

    print('Training completed. Starting inference / trajectory trace...')

    if ckpt_path.exists():
        model.load_state_dict(safe_torch_load(ckpt_path, device))
    model.eval()

    # Trace a single sample trajectory
    x = torch.randn(1, 2, device=device)
    trajectory = [x.detach().cpu().numpy()]

    with torch.no_grad():
        for t in reversed(range(n_steps)):
            t_tensor = torch.tensor([t], device=device)
            predicted_noise = model(x, t_tensor)
            x = x - predicted_noise * 0.1
            trajectory.append(x.detach().cpu().numpy())

    final_xy = x.detach().cpu().numpy()
    print(f'Generated coordinate: {final_xy}')

    visualize_trajectory(trajectory, plot_path)
    print(f'Plot saved: {plot_path.name}')

    meta = {
        'model_type': 'SimpleDiffusion(2D demo)',
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'seed': seed,
        'n_steps': n_steps,
        'best_loss': best_loss,
        'checkpoint': str(ckpt_path.name),
        'plot': str(plot_path.name),
    }
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(f'Metadata written: {meta_path.name}')
    except Exception as e:
        print(f'[Warning] Failed to write metadata: {type(e).__name__}: {e}')

    print('Done.')


if __name__ == '__main__':
    main()
