"""Бенчмарки і графіки для аналізу результатів.

Збирає метрики по серіях запусків і будує графіки в `out/`:

  1. perf_vs_n.png         — час vs N частинок (safe_per_cell vs pooled).
  2. perf_vs_workers.png   — час vs кількість воркерів у pooled.
  3. grid_size_effect.png  — час vs розмір сітки (контенція на локи).
  4. density_effect.png    — час і коректність vs щільність частинок.
  5. drift_demo.png        — приклад симуляції з p_right > p_left (видимий дрейф).

Також генерує `out/benchmark_summary.txt` з числовими результатами.
"""

from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import SimulationConfig
from simulation import build_simulation


@dataclass
class BenchPoint:
    label: str
    x: float
    wall_time_s: float
    initial: int
    final: int

    @property
    def conserved(self) -> bool:
        return self.initial == self.final


def _run_silent(cfg: SimulationConfig) -> tuple[float, int, int]:
    """Запустити симуляцію без логу і повернути (wall, initial, final)."""
    cfg = dataclasses.replace(cfg, verbose=False)
    sim = build_simulation(cfg)
    result = sim.run()
    return result.wall_time_s, result.initial_total, result.final_total


def _ensure_out(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def bench_perf_vs_n(
    out_dir: str = "out",
    n_values_safe: Sequence[int] = (200, 500, 1000, 2000, 4000, 8000),
    n_values_pooled: Sequence[int] = (200, 500, 1000, 2000, 4000, 10000, 20000, 50000),
    base_cfg: Optional[SimulationConfig] = None,
    n_workers: int = 8,
) -> List[BenchPoint]:
    """Час симуляції vs N (для safe_per_cell і pooled)."""
    print("\n=== BENCH: perf vs N (1 thread per atom vs pooled) ===")
    base = base_cfg or SimulationConfig(
        width=80, height=80, total_steps=200, snapshot_every=200, seed=42, verbose=False,
    )

    points: list[BenchPoint] = []

    # Якщо ОС відмовилась створювати нові потоки, більші N теж не підуть.
    for n in n_values_safe:
        cfg = dataclasses.replace(base, mode="safe_per_cell", n_particles=n)
        try:
            wall, ini, fin = _run_silent(cfg)
            points.append(BenchPoint("safe_per_cell", n, wall, ini, fin))
            print(f"  safe_per_cell  N={n:>6} → {wall:6.2f}s  (init={ini}, final={fin})")
        except Exception as exc:
            print(
                f"  safe_per_cell  N={n:>6} → FAILED: {exc!r}  "
                "← ОС-ліміт потоків ('1 потік на атом' не масштабується)"
            )
            break

    for n in n_values_pooled:
        cfg = dataclasses.replace(
            base, mode="pooled", n_particles=n, n_workers=n_workers,
        )
        try:
            wall, ini, fin = _run_silent(cfg)
            points.append(BenchPoint("pooled", n, wall, ini, fin))
            print(f"  pooled         N={n:>6} → {wall:6.2f}s  (init={ini}, final={fin})")
        except Exception as exc:
            print(f"  pooled         N={n:>6} → FAILED: {exc!r}")

    _plot_perf_vs_n(points, os.path.join(out_dir, "perf_vs_n.png"))
    return points


def _plot_perf_vs_n(points: Sequence[BenchPoint], out_path: str) -> None:
    _ensure_out(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, marker, color in [
        ("safe_per_cell", "o", "tab:red"),
        ("pooled", "s", "tab:blue"),
    ]:
        xs = [p.x for p in points if p.label == label]
        ys = [p.wall_time_s for p in points if p.label == label]
        if not xs:
            continue
        ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (кількість частинок)")
    ax.set_ylabel("Час симуляції, с (log)")
    ax.set_title("Масштабування: час vs кількість частинок\n"
                 "safe_per_cell (1 потік на атом) vs pooled (ThreadPoolExecutor)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ANL]    saved {out_path}")


def bench_perf_vs_workers(
    out_dir: str = "out",
    workers_grid: Sequence[int] = (1, 2, 4, 8, 16, 32),
    n_particles: int = 5000,
    base_cfg: Optional[SimulationConfig] = None,
) -> List[BenchPoint]:
    """Час pooled-симуляції vs кількість воркерів."""
    print("\n=== BENCH: perf vs workers (pooled) ===")
    base = base_cfg or SimulationConfig(
        width=80, height=80, total_steps=200,
        snapshot_every=200, seed=42, verbose=False,
    )
    points: list[BenchPoint] = []
    for w in workers_grid:
        cfg = dataclasses.replace(
            base, mode="pooled", n_particles=n_particles, n_workers=w,
        )
        wall, ini, fin = _run_silent(cfg)
        points.append(BenchPoint("pooled", w, wall, ini, fin))
        print(f"  workers={w:>3} → {wall:6.2f}s  (init={ini}, final={fin})")

    _plot_perf_vs_workers(points, os.path.join(out_dir, "perf_vs_workers.png"), n_particles)
    return points


def _plot_perf_vs_workers(points: Sequence[BenchPoint], out_path: str, n: int) -> None:
    _ensure_out(os.path.dirname(out_path))
    xs = [p.x for p in points]
    ys = [p.wall_time_s for p in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "-o", color="tab:purple", linewidth=2)
    ax.set_xlabel("Кількість воркерів у ThreadPoolExecutor")
    ax.set_ylabel("Час симуляції, с")
    ax.set_title(f"Pooled: час vs кількість воркерів (N={n})")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ANL]    saved {out_path}")


def bench_grid_size_effect(
    out_dir: str = "out",
    sizes: Sequence[int] = (10, 20, 40, 80, 160),
    n_particles: int = 2000,
    base_cfg: Optional[SimulationConfig] = None,
) -> List[BenchPoint]:
    """Час pooled vs розмір сітки (контенція на per-cell локи)."""
    print("\n=== BENCH: perf vs grid size (pooled) ===")
    base = base_cfg or SimulationConfig(
        total_steps=200, snapshot_every=200, seed=42, verbose=False,
    )
    points: list[BenchPoint] = []
    for s in sizes:
        cfg = dataclasses.replace(
            base, mode="pooled", width=s, height=s,
            n_particles=n_particles, n_workers=8,
        )
        wall, ini, fin = _run_silent(cfg)
        points.append(BenchPoint(f"pooled-{s}x{s}", s, wall, ini, fin))
        density = n_particles / (s * s)
        print(f"  grid={s:>3}x{s:<3} (density={density:.2f}) → {wall:6.2f}s")

    _plot_grid_size_effect(points, os.path.join(out_dir, "grid_size_effect.png"), n_particles)
    return points


def _plot_grid_size_effect(points: Sequence[BenchPoint], out_path: str, n: int) -> None:
    _ensure_out(os.path.dirname(out_path))
    xs = [p.x for p in points]
    ys = [p.wall_time_s for p in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "-o", color="tab:green", linewidth=2)
    ax.set_xlabel("Сторона сітки H = W")
    ax.set_ylabel("Час симуляції, с")
    ax.set_title(f"Контенція на локи: час vs розмір сітки (N={n})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ANL]    saved {out_path}")


def bench_density_effect(
    out_dir: str = "out",
    densities: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0),
    grid_size: int = 40,
    base_cfg: Optional[SimulationConfig] = None,
) -> List[BenchPoint]:
    """Вплив щільності на час та коректність (pooled)."""
    print("\n=== BENCH: density effect (pooled) ===")
    base = base_cfg or SimulationConfig(
        total_steps=200, snapshot_every=200, seed=42, verbose=False,
    )
    points: list[BenchPoint] = []
    for d in densities:
        n = max(1, int(d * grid_size * grid_size))
        cfg = dataclasses.replace(
            base, mode="pooled", width=grid_size, height=grid_size,
            n_particles=n, n_workers=8,
        )
        wall, ini, fin = _run_silent(cfg)
        points.append(BenchPoint(f"density={d}", d, wall, ini, fin))
        print(f"  density={d:>5} → N={n:>5}  time={wall:6.2f}s  conserved={ini==fin}")

    _plot_density_effect(points, os.path.join(out_dir, "density_effect.png"), grid_size)
    return points


def _plot_density_effect(points: Sequence[BenchPoint], out_path: str, grid: int) -> None:
    _ensure_out(os.path.dirname(out_path))
    xs = [p.x for p in points]
    ys = [p.wall_time_s for p in points]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "-o", color="tab:orange", linewidth=2)
    ax.set_xlabel("Щільність N / (H·W)")
    ax.set_ylabel("Час симуляції, с")
    ax.set_title(f"Вплив щільності частинок на час (сітка {grid}×{grid})")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[ANL]    saved {out_path}")


def bench_diffusion_law(out_dir: str = "out") -> dict:
    """Демо закону Ейнштейна: ⟨r²⟩ ∝ t (концентрована початкова умова).

    Усі частинки стартують у центральній клітинці, симетричні ймовірності
    переходу. Очікуємо приблизно лінійне зростання Var(x) + Var(y) у часі —
    це добра перевірка дифузійної поведінки моделі.

    Зберігає `diffusion_law.png` (з лінійною підгонкою).
    """
    print("\n=== DEMO: diffusion law (⟨r²⟩ ∝ t) ===")
    cfg = SimulationConfig(
        width=120, height=120, n_particles=2000,
        p_up=0.25, p_down=0.25, p_left=0.25, p_right=0.25,
        total_steps=400, snapshot_every=10, seed=42, mode="pooled",
        n_workers=8, initial_layout="concentrate", verbose=False,
    )
    sim = build_simulation(cfg)
    result = sim.run()
    print(f"  initial={result.initial_total}, final={result.final_total}, "
          f"time={result.wall_time_s:.2f}s")

    steps = []
    var_total = []
    for f in result.frames:
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
        var_total.append(vx + vy)

    out_path = os.path.join(out_dir, "diffusion_law.png")
    _ensure_out(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, var_total, "-o", color="tab:blue", label="⟨r²⟩ = Var(x)+Var(y)")
    if len(steps) >= 2:
        slope, intercept = np.polyfit(steps[1:], var_total[1:], 1)
        ax.plot(steps, [slope * s + intercept for s in steps],
                "--", color="tab:red",
                label=f"лінійна підгонка: ⟨r²⟩ ≈ {slope:.3f}·t {intercept:+.2f}")
    ax.set_xlabel("Крок симуляції t")
    ax.set_ylabel("⟨r²⟩ (середній квадрат зміщення)")
    ax.set_title("Закон Ейнштейна: ⟨r²⟩ ∝ t (старт із концентрованої початкової умови)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[VIS]    saved figure: {out_path}")

    # Heatmap для звіту: видно радіальне розповзання.
    from visualization import plot_initial_vs_final
    plot_initial_vs_final(
        result.frames,
        os.path.join(out_dir, "diffusion_initial_vs_final.png"),
        title="Дифузія: усі частинки стартують у центрі, симетричні ймовірності",
        share_scale=False,
    )

    return {
        "slope": float(slope) if len(steps) >= 2 else 0.0,
        "intercept": float(intercept) if len(steps) >= 2 else 0.0,
        "n_frames": len(steps),
    }


def bench_drift_demo(out_dir: str = "out") -> tuple[list, dict]:
    """Запустити симуляцію з зсувом ймовірностей → видимий дрейф.

    Зберігає `drift_demo.png` (initial vs final) та повертає кадри + метадані.
    Параметри підібрані так, щоб дрейф був чітко помітний на heatmap, але
    не зливався в одну точку (тоді колірна шкала розчавлюється).
    """
    print("\n=== DEMO: drift (p_right > p_left, p_down > p_up) ===")
    cfg = SimulationConfig(
        width=60, height=60, n_particles=2500,
        # Помірний зсув: 0.3 vs 0.2 дає помітний дрейф.
        p_up=0.20, p_down=0.30, p_left=0.20, p_right=0.30,
        total_steps=150, snapshot_every=150, seed=7, mode="pooled",
        n_workers=8, verbose=False,
    )
    sim = build_simulation(cfg)
    result = sim.run()
    print(f"  initial={result.initial_total}, final={result.final_total}, "
          f"time={result.wall_time_s:.2f}s")

    from visualization import plot_initial_vs_final  # local import to avoid cycles

    out_path = os.path.join(out_dir, "drift_demo.png")
    plot_initial_vs_final(
        result.frames, out_path,
        title="Демонстрація дрейфу: p_right=p_down=0.30 > p_left=p_up=0.20",
        share_scale=False,
    )
    return result.frames, {"initial": result.initial_total, "final": result.final_total}


def demo_race_condition(out_dir: str = "out") -> dict:
    """Запустити unsafe + safe для демонстрації race condition."""
    print("\n=== DEMO: race condition ===")
    cfg = SimulationConfig(
        width=8, height=8, n_particles=600, total_steps=600,
        snapshot_every=600, seed=123, mode="unsafe", verbose=False,
    )
    unsafe_wall, unsafe_init, unsafe_final = _run_silent(cfg)
    print(f"  UNSAFE: init={unsafe_init} final={unsafe_final} "
          f"diff={unsafe_final - unsafe_init} time={unsafe_wall:.2f}s")

    cfg2 = dataclasses.replace(cfg, mode="safe_per_cell")
    safe_wall, safe_init, safe_final = _run_silent(cfg2)
    print(f"  SAFE  : init={safe_init} final={safe_final} "
          f"diff={safe_final - safe_init} time={safe_wall:.2f}s")

    return {
        "unsafe": (unsafe_init, unsafe_final, unsafe_wall),
        "safe": (safe_init, safe_final, safe_wall),
    }


def demo_deadlock(out_dir: str = "out") -> dict:
    """Спроба deadlock_demo + перевірка deadlock_fixed."""
    print("\n=== DEMO: deadlock ===")
    cfg = SimulationConfig(
        width=4, height=4, n_particles=80, total_steps=300, snapshot_every=300,
        seed=11, mode="deadlock_demo", deadlock_timeout=5.0, verbose=False,
    )
    sim = build_simulation(cfg)
    result_demo = sim.run()
    print(f"  deadlock_demo: detected={result_demo.deadlock_detected} "
          f"time={result_demo.wall_time_s:.2f}s "
          f"diff={result_demo.diff:+d}")

    cfg2 = dataclasses.replace(cfg, mode="deadlock_fixed", verbose=False)
    sim2 = build_simulation(cfg2)
    result_fixed = sim2.run()
    print(f"  deadlock_fixed: detected={result_fixed.deadlock_detected} "
          f"time={result_fixed.wall_time_s:.2f}s "
          f"diff={result_fixed.diff:+d}")

    return {
        "demo": (result_demo.deadlock_detected, result_demo.wall_time_s, result_demo.diff),
        "fixed": (result_fixed.deadlock_detected, result_fixed.wall_time_s, result_fixed.diff),
    }


def run_full_benchmark(out_dir: str = "out") -> None:
    """Прогнати всі бенчмарки і скласти текстовий summary."""
    _ensure_out(out_dir)
    summary_lines: list[str] = []

    def _log(line: str) -> None:
        print(line)
        summary_lines.append(line)

    _log("="*70)
    _log("ПОВНИЙ БЕНЧМАРК — Броунівський рух")
    _log(f"Час запуску: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _log("="*70)

    pts_n = bench_perf_vs_n(out_dir=out_dir)
    _log("\n[perf_vs_n]")
    for p in pts_n:
        _log(f"  {p.label:>14}  N={p.x:>6}  t={p.wall_time_s:6.2f}s  "
             f"conserved={p.conserved}")

    pts_w = bench_perf_vs_workers(out_dir=out_dir)
    _log("\n[perf_vs_workers]")
    for p in pts_w:
        _log(f"  workers={int(p.x):>3}  t={p.wall_time_s:6.2f}s")

    pts_g = bench_grid_size_effect(out_dir=out_dir)
    _log("\n[grid_size_effect]")
    for p in pts_g:
        _log(f"  grid={int(p.x):>3}x{int(p.x):<3}  t={p.wall_time_s:6.2f}s")

    pts_d = bench_density_effect(out_dir=out_dir)
    _log("\n[density_effect]")
    for p in pts_d:
        _log(f"  density={p.x:>5}  t={p.wall_time_s:6.2f}s  conserved={p.conserved}")

    bench_drift_demo(out_dir=out_dir)

    diff_info = bench_diffusion_law(out_dir=out_dir)
    _log("\n[diffusion_law]")
    _log(f"  лінійна підгонка ⟨r²⟩ vs t: slope={diff_info['slope']:.4f}, "
         f"intercept={diff_info['intercept']:.2f}")

    race = demo_race_condition(out_dir=out_dir)
    u_init, u_final, u_t = race["unsafe"]
    s_init, s_final, s_t = race["safe"]
    _log("\n[race_condition_demo]")
    _log(f"  UNSAFE  : initial={u_init} final={u_final} diff={u_final-u_init:+d} "
         f"  ← очікувано: diff < 0 (втрата частинок)")
    _log(f"  SAFE    : initial={s_init} final={s_final} diff={s_final-s_init:+d} "
         f"  ← очікувано: diff = 0 (інваріант збережено)")

    dl = demo_deadlock(out_dir=out_dir)
    demo_det, demo_t, demo_diff = dl["demo"]
    fix_det, fix_t, fix_diff = dl["fixed"]
    _log("\n[deadlock_demo]")
    _log(f"  deadlock_demo : detected={demo_det} time={demo_t:.2f}s diff={demo_diff:+d}")
    _log(f"  deadlock_fixed: detected={fix_det} time={fix_t:.2f}s diff={fix_diff:+d}")

    _log("\n[SUMMARY]")
    _log("  Графіки збережено у out/:")
    for fname in (
        "perf_vs_n.png", "perf_vs_workers.png", "grid_size_effect.png",
        "density_effect.png", "drift_demo.png", "diffusion_law.png",
        "diffusion_initial_vs_final.png",
    ):
        _log(f"    - out/{fname}")

    summary_path = os.path.join(out_dir, "benchmark_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))
    print(f"\n[ANL]    summary saved → {summary_path}")
