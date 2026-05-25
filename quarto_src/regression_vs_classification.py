import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

x = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-x))

# Regression data: sigmoid + noise, continuous targets
x_reg = rng.uniform(-6, 6, 60)
y_reg = np.clip(1 / (1 + np.exp(-x_reg)) + rng.normal(0, 0.07, 60), 0, 1)

# Classification data: same x, threshold at 0.5 -> binary labels
x_cls = rng.uniform(-6, 6, 60)
prob_cls = 1 / (1 + np.exp(-x_cls))
y_cls = (prob_cls > 0.5).astype(float)

threshold_x = 0.0   # sigmoid(0) = 0.5
threshold_y = 0.5

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

# --- Left: Regression ---
ax = axes[0]
ax.scatter(x_reg, y_reg, alpha=0.6, color="steelblue", s=25, zorder=3, label="Data points")
ax.plot(x, sigmoid, "k--", linewidth=1.8, label="Sigmoid model")
ax.set_title("Regression")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(-6, 6)
ax.set_ylim(-0.15, 1.15)
ax.legend()

# --- Right: Classification ---
ax = axes[1]
ax.scatter(x_cls[y_cls == 0], y_cls[y_cls == 0], alpha=0.7, color="tomato",
           s=30, zorder=3, label="Class 0")
ax.scatter(x_cls[y_cls == 1], y_cls[y_cls == 1], alpha=0.7, color="steelblue",
           s=30, zorder=3, label="Class 1")
ax.plot(x, sigmoid, "k--", linewidth=1.8, label="Sigmoid model")

# Threshold crosshair
ax.axvline(threshold_x, color="gray", linewidth=1.2, linestyle="-")
ax.axhline(threshold_y, color="gray", linewidth=1.2, linestyle="-")
ax.annotate("threshold\n(p = 0.5)", xy=(threshold_x, threshold_y),
            xytext=(1.5, 0.38), fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

ax.set_title("Classification")
ax.set_xlabel("x")
ax.set_ylabel("class label")
ax.set_xlim(-6, 6)
ax.set_ylim(-0.15, 1.15)
ax.set_yticks([0, 1])
ax.legend()

fig.tight_layout()
plt.savefig("regression_vs_classification.png", dpi=150, bbox_inches="tight")
plt.show()
