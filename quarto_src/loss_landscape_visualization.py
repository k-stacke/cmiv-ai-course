"""
2-D loss landscape visualization using two random orthogonal directions.

Two orthogonal unit vectors d1, d2 define a plane through the initial weights
theta_0. The landscape shows loss(theta_0 + a1*d1 + a2*d2) on the current
minibatch over a fixed (a1, a2) grid. The red dot marks where the current
weights project onto this plane. All axes are globally fixed.

Two-pass strategy
-----------------
Pass 1 — train, save each minibatch's (xb, yb) + projection (a1_t, a2_t).
Pass 2 — once the global grid is known, sweep the grid for every snapshot.
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE = 1729
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ── Dataset (same as notebook: moons, 500 samples) ─────────────────────────────
X, y = datasets.make_moons(n_samples=500, noise=0.2, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
sc = StandardScaler().fit(X_train)
Xt_train = torch.tensor(sc.transform(X_train), dtype=torch.float32)
yt_train = torch.tensor(y_train, dtype=torch.float32)

# ── Model (same as notebook: MLP [16, 16]) ─────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden_sizes: list[int] = [16, 16]):
        super().__init__()
        layers = []
        in_size = 2
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Parameter-space helpers ────────────────────────────────────────────────────
def get_params_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_params_flat(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view(p.shape))
        idx += n


# ── Training setup ─────────────────────────────────────────────────────────────
model = MLP([16, 16])
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.BCEWithLogitsLoss()
train_loader = DataLoader(
    TensorDataset(Xt_train, yt_train), batch_size=32, shuffle=True
)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

N_EPOCHS = 30
N_GRID = 50          # grid resolution per axis; increase for finer contours
GRID_PAD = 0.5       # extra margin around the trajectory bounding box

# ── Pass 1: train and collect snapshots ───────────────────────────────────────
snapshots = []
minibatch_count = 0

print("Pass 1 — training...")
for epoch in range(N_EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        minibatch_count += 1

        model.eval()
        snapshots.append({
            "minibatch": minibatch_count,
            "epoch":     epoch + 1,
            "xb":        xb.clone(),
            "yb":        yb.clone(),
            "theta":     get_params_flat(model).clone(),
        })
        model.train()

theta_final = get_params_flat(model).clone()
print(f"  {len(snapshots)} snapshots collected")

# ── Compute Hessian at theta_final (sharpest-curvature directions) ─────────────
print("Computing Hessian at final weights...")
set_params_flat(model, theta_final)
model.zero_grad()

loss_full = criterion(model(Xt_train), yt_train)
grads_1st = torch.autograd.grad(loss_full, list(model.parameters()), create_graph=True)
grad_flat = torch.cat([g.reshape(-1) for g in grads_1st])

H = torch.zeros(n_params, n_params)
for i in range(n_params):
    grads_2nd = torch.autograd.grad(grad_flat[i], list(model.parameters()), retain_graph=True)
    H[i] = torch.cat([g.reshape(-1) for g in grads_2nd]).detach()

# eigh returns eigenvalues sorted ascending; take last two = sharpest directions
eigenvalues, eigenvectors = torch.linalg.eigh(H)
d1 = eigenvectors[:, -1]   # largest eigenvalue
d2 = eigenvectors[:, -2]   # second largest eigenvalue
print(f"  Top-2 eigenvalues: {eigenvalues[-1]:.4f}, {eigenvalues[-2]:.4f}")
print(f"  Min eigenvalue: {eigenvalues[0]:.4f}")

# Reference origin: final weights (trajectory converges toward (0, 0))
theta_ref = theta_final

# Project each snapshot onto the Hessian eigenvector plane
for snap in snapshots:
    delta = snap["theta"] - theta_ref
    snap["alpha1"] = torch.dot(delta, d1).item()
    snap["alpha2"] = torch.dot(delta, d2).item()

# ── Build fixed 2-D grid covering the full trajectory ─────────────────────────
a1_vals = [s["alpha1"] for s in snapshots]
a2_vals = [s["alpha2"] for s in snapshots]
a1_grid = np.linspace(min(a1_vals) - GRID_PAD, max(a1_vals) + GRID_PAD, N_GRID)
a2_grid = np.linspace(min(a2_vals) - GRID_PAD, max(a2_vals) + GRID_PAD, N_GRID)
print(f"  a1 range: [{a1_grid[0]:.3f}, {a1_grid[-1]:.3f}]")
print(f"  a2 range: [{a2_grid[0]:.3f}, {a2_grid[-1]:.3f}]")

# ── Pass 2: evaluate 2-D loss landscapes ──────────────────────────────────────
print(f"Pass 2 — evaluating {len(snapshots)} × {N_GRID}×{N_GRID} landscapes...")
for idx, snap in enumerate(snapshots):
    if idx % 100 == 0:
        print(f"  {idx}/{len(snapshots)}")
    z = np.empty((N_GRID, N_GRID))   # z[row=a2, col=a1]
    xb, yb = snap["xb"], snap["yb"]
    with torch.no_grad():
        for i, a1 in enumerate(a1_grid):
            theta_base = theta_ref + float(a1) * d1
            for j, a2 in enumerate(a2_grid):
                set_params_flat(model, theta_base + float(a2) * d2)
                z[j, i] = criterion(model(xb), yb).item()
    snap["z"] = z

# Restore final trained weights (good hygiene)
set_params_flat(model, theta_final)

# ── Global z range (fixed across all frames) ───────────────────────────────────
z_min = min(snap["z"].min() for snap in snapshots)
z_max = max(snap["z"].max() for snap in snapshots)
print(f"  Global z range: [{z_min:.4f}, {z_max:.4f}]")

# ── Save for Manim renderer ────────────────────────────────────────────────────
out_path = "loss_landscape_data.npz"
np.savez(
    out_path,
    a1_grid=a1_grid,
    a2_grid=a2_grid,
    z_min=z_min,
    z_max=z_max,
    z_frames=np.stack([snap["z"] for snap in snapshots]),          # (N, H, W)
    alpha1=np.array([snap["alpha1"] for snap in snapshots]),
    alpha2=np.array([snap["alpha2"] for snap in snapshots]),
    epochs=np.array([snap["epoch"] for snap in snapshots]),
    minibatches=np.array([snap["minibatch"] for snap in snapshots]),
)
print(f"Saved → {out_path}")
