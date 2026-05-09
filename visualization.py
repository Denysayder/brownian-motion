"""Графіки та анімації для результатів симуляції."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # працюємо без GUI, одразу пишемо у файли
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from snapshot import Snapshot


def animate_simulation(
    frames: Sequence[Snapshot],
    out_path: str,
    title: str = "Броунівський рух у 2D-кристалі",
    fps: int = 8,
    cmap: str = "viridis",
) -> str:
    """Зберегти heatmap-анімацію у MP4 або GIF."""
    if not frames:
        raise ValueError("Список frames пустий — нема що анімувати")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Спільний vmax не дає кольоровій шкалі стрибати між кадрами.
    vmax = max(int(f.cells.max()) for f in frames) or 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        frames[0].cells,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Частинок у клітинці")
    title_text = ax.set_title(_frame_title(title, frames[0]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i):
        f = frames[i]
        im.set_data(f.cells)
        title_text.set_text(_frame_title(title, f))
        return [im, title_text]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=max(1, int(1000 / fps)),
        blit=False,
        repeat=False,
    )

    saved_path = _save_animation(ani, out_path, fps=fps)
    plt.close(fig)
    return saved_path


def _frame_title(base: str, snap: Snapshot) -> str:
    return f"{base}\nКрок {snap.step} | Усього частинок: {snap.total}"


def _save_animation(ani: FuncAnimation, out_path: str, fps: int) -> str:
    """Спробувати MP4 через ffmpeg, а якщо його нема — GIF через Pillow."""
    base, _ = os.path.splitext(out_path)
    try:
        from matplotlib.animation import FFMpegWriter

        if FFMpegWriter.isAvailable():
            mp4_path = base + ".mp4"
            ani.save(mp4_path, writer=FFMpegWriter(fps=fps, bitrate=1800))
            print(f"[VIS]    saved animation: {mp4_path}")
            return mp4_path
    except Exception as exc:
        print(f"[VIS]    MP4 writer недоступний: {exc}; fallback to GIF")

    gif_path = base + ".gif"
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    print(f"[VIS]    saved animation: {gif_path}")
    return gif_path


def plot_initial_vs_final(
    frames: Sequence[Snapshot],
    out_path: str,
    title: str = "Розподіл частинок: до та після симуляції",
    share_scale: bool = False,
) -> None:
    """Порівняти початковий і фінальний розподіли на двох heatmap."""
    if len(frames) < 2:
        raise ValueError("Потрібно ≥ 2 кадри (initial + final)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    initial, final = frames[0], frames[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if share_scale:
        vmax = max(int(initial.cells.max()), int(final.cells.max())) or 1
        vmaxes = [vmax, vmax]
    else:
        vmaxes = [
            max(1, int(initial.cells.max())),
            max(1, int(final.cells.max())),
        ]

    for ax, snap, label, vmx in zip(
        axes,
        (initial, final),
        ("Початок (step=0)", f"Кінець (step={final.step})"),
        vmaxes,
    ):
        im = ax.imshow(
            snap.cells, cmap="viridis", vmin=0, vmax=vmx,
            origin="lower", interpolation="nearest",
        )
        ax.set_title(f"{label}\nУсього: {snap.total} | max у клітинці: {int(snap.cells.max())}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[VIS]    saved figure: {out_path}")


def plot_total_count(
    frames: Sequence[Snapshot],
    out_path: str,
    expected: Optional[int] = None,
    title: str = "Інваріант збереження: загальна кількість частинок vs крок",
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    steps = [f.step for f in frames]
    totals = [f.total for f in frames]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, totals, "-o", color="tab:blue", label="Виміряно")
    if expected is not None:
        ax.axhline(expected, color="tab:red", linestyle="--", label=f"Очікувано = {expected}")
    ax.set_xlabel("Крок симуляції")
    ax.set_ylabel("Загальна кількість частинок")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[VIS]    saved figure: {out_path}")


def plot_displacement_variance(
    frames: Sequence[Snapshot],
    out_path: str,
    title: str = "Дисперсія розподілу частинок vs крок (дифузія)",
) -> None:
    """Побудувати Var(x), Var(y) і їхню суму в часі."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    steps, var_x, var_y, var_total = [], [], [], []
    for f in frames:
        cells = f.cells.astype(np.float64)
        total = cells.sum()
        if total <= 0:
            continue
        H, W = cells.shape
        ys = np.arange(H).reshape(-1, 1)
        xs = np.arange(W).reshape(1, -1)
        mean_x = (cells * xs).sum() / total
        mean_y = (cells * ys).sum() / total
        vx = ((cells * (xs - mean_x) ** 2).sum()) / total
        vy = ((cells * (ys - mean_y) ** 2).sum()) / total
        steps.append(f.step)
        var_x.append(vx)
        var_y.append(vy)
        var_total.append(vx + vy)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, var_x, "-", label="Var(x)")
    ax.plot(steps, var_y, "-", label="Var(y)")
    ax.plot(steps, var_total, "--", label="Var(x)+Var(y)", linewidth=2)
    ax.set_xlabel("Крок симуляції")
    ax.set_ylabel("Дисперсія координати")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[VIS]    saved figure: {out_path}")
