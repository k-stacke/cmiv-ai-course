"""
Manim renderer for the 2-D loss landscape animation.

Run computation first (default pixi env):
    pixi run python quarto_src/loss_landscape_visualization.py

Then render (manim pixi env):
    pixi run -e manim manim -pql quarto_src/loss_landscape_manim.py LossLandscape
    pixi run -e manim manim -pqh quarto_src/loss_landscape_manim.py LossLandscape
"""

from pathlib import Path

import numpy as np
import matplotlib.cm as cm
from manim import (
    Scene, Axes, ImageMobject, Dot, Text, VGroup,
    UpdateFromAlphaFunc, WHITE, RED, GREY_B, LEFT, RIGHT, DOWN, UP,
    linear,
)

# ── Load pre-computed data ─────────────────────────────────────────────────────
_data_path = Path(__file__).parent / "loss_landscape_data.npz"
data = np.load(_data_path)

a1_grid    = data["a1_grid"]
a2_grid    = data["a2_grid"]
z_min      = float(data["z_min"])
z_max      = float(data["z_max"])
z_frames   = data["z_frames"]        # (N, H, W)
alpha1s    = data["alpha1"]
alpha2s    = data["alpha2"]
epochs     = data["epochs"]
minibatches = data["minibatches"]

N_FRAMES = len(z_frames)
FPS = 30  # minibatches per second of video


def _to_rgba(z: np.ndarray) -> np.ndarray:
    """(H, W) float → (H, W, 4) uint8 using viridis colormap."""
    norm = (z - z_min) / max(z_max - z_min, 1e-8)
    return (cm.viridis(norm) * 255).astype(np.uint8)


# Pre-render all RGBA frames so the scene update is just an array copy.
print("Pre-rendering RGBA frames...")
rgba_frames = [_to_rgba(z_frames[i]) for i in range(N_FRAMES)]
print(f"  {N_FRAMES} frames ready")


class LossLandscape(Scene):
    def construct(self):
        x_span = float(a1_grid[-1] - a1_grid[0])
        y_span = float(a2_grid[-1] - a2_grid[0])
        aspect = y_span / x_span

        # Axes sized to fill most of the scene
        axes_w = 5.5
        axes_h = axes_w * aspect

        x_step = round(x_span / 4, 1) or 0.5
        y_step = round(y_span / 4, 1) or 0.5

        axes = Axes(
            x_range=[float(a1_grid[0]), float(a1_grid[-1]), x_step],
            y_range=[float(a2_grid[0]), float(a2_grid[-1]), y_step],
            x_length=axes_w,
            y_length=axes_h,
            tips=False,
            axis_config={"color": WHITE, "include_numbers": True,
                         "decimal_number_config": {"num_decimal_places": 1}},
        )
        x_label = axes.get_x_axis_label(r"\alpha_1", direction=DOWN)
        y_label = axes.get_y_axis_label(r"\alpha_2", direction=LEFT)
        axes_group = VGroup(axes, x_label, y_label).center().shift(DOWN * 0.3)

        # Heatmap image aligned to axes coordinate area
        bl = axes.c2p(float(a1_grid[0]),  float(a2_grid[0]))
        tr = axes.c2p(float(a1_grid[-1]), float(a2_grid[-1]))
        img_w = abs(tr[0] - bl[0])
        img_h = abs(tr[1] - bl[1])
        img_center = np.array([(bl[0] + tr[0]) / 2,
                                (bl[1] + tr[1]) / 2, 0.0])

        image = ImageMobject(rgba_frames[0])
        image.set_width(img_w).set_height(img_h).move_to(img_center)

        # Dot at projected position
        dot = Dot(
            point=axes.c2p(float(alpha1s[0]), float(alpha2s[0])),
            color=RED, radius=0.10,
        ).set_stroke(WHITE, width=1.5)

        # Labels
        title = Text("Loss Landscape – Top-2 Hessian Eigenvectors",
                     font_size=22).to_edge(UP)
        info = Text(
            f"Epoch {epochs[0]:3d}  batch {minibatches[0]:4d}",
            font_size=16, color=GREY_B,
        ).to_corner(RIGHT + DOWN)

        self.add(image, axes_group, dot, title, info)

        duration = N_FRAMES / FPS

        def update_image(mob, alpha):
            idx = min(int(alpha * N_FRAMES), N_FRAMES - 1)
            mob.pixel_array = rgba_frames[idx]

        def update_dot(mob, alpha):
            idx = min(int(alpha * N_FRAMES), N_FRAMES - 1)
            mob.move_to(axes.c2p(float(alpha1s[idx]), float(alpha2s[idx])))

        def update_info(mob, alpha):
            idx = min(int(alpha * N_FRAMES), N_FRAMES - 1)
            mob.become(
                Text(
                    f"Epoch {epochs[idx]:3d}  batch {minibatches[idx]:4d}",
                    font_size=16, color=GREY_B,
                ).to_corner(RIGHT + DOWN)
            )

        self.play(
            UpdateFromAlphaFunc(image, update_image),
            UpdateFromAlphaFunc(dot,   update_dot),
            UpdateFromAlphaFunc(info,  update_info),
            run_time=duration,
            rate_func=linear,
        )
        self.wait(1)
