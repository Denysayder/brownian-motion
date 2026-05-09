"""CLI для запуску симуляцій броунівського руху.

Приклади:

    # базовий безпечний режим (1 потік = 1 атом)
    python main.py --mode safe_per_cell --width 50 --height 50 --particles 500 \\
        --steps 1000 --snapshot-every 50 --seed 42

    # демо race condition
    python main.py --mode unsafe --width 8 --height 8 --particles 600 --steps 600

    # демо deadlock (з тайм-аутом)
    python main.py --mode deadlock_demo --width 4 --height 4 --particles 80 \\
        --steps 300 --deadlock-timeout 5

    # фікс deadlock
    python main.py --mode deadlock_fixed --width 4 --height 4 --particles 80 \\
        --steps 300

    # масштабований режим
    python main.py --mode pooled --particles 100000 --steps 200 --workers 8

    # повний бенчмарк-набір (для report.md)
    python main.py --benchmark
"""

from __future__ import annotations

import argparse
import os
import sys

# Щоб `python main.py` працював без встановлення пакета.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import SimulationConfig
from simulation import build_simulation
from visualization import (
    animate_simulation,
    plot_displacement_variance,
    plot_initial_vs_final,
    plot_total_count,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Моделювання броунівського руху домішки у 2D-кристалі "
            "(навчальний проект 1 МТПП 2026)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--mode",
        choices=["safe_per_cell", "unsafe", "deadlock_demo", "deadlock_fixed", "pooled"],
        default="safe_per_cell",
        help="Режим симуляції (default: safe_per_cell — 1 потік = 1 атом).",
    )
    p.add_argument("--width", type=int, default=50, help="Ширина сітки кристалу.")
    p.add_argument("--height", type=int, default=50, help="Висота сітки кристалу.")
    p.add_argument("--particles", type=int, default=500, help="Кількість атомів домішки.")

    p.add_argument("--p-up", type=float, default=0.25)
    p.add_argument("--p-down", type=float, default=0.25)
    p.add_argument("--p-left", type=float, default=0.25)
    p.add_argument("--p-right", type=float, default=0.25)

    p.add_argument("--steps", type=int, default=1000, help="Загальна кількість кроків.")
    p.add_argument("--snapshot-every", type=int, default=50,
                   help="Дискретизація знімків стану.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8,
                   help="Кількість воркерів для pooled-режиму.")
    p.add_argument("--deadlock-timeout", type=float, default=15.0,
                   help="Тайм-аут вотчдога для deadlock_demo (с).")

    p.add_argument("--initial-layout", choices=["scatter", "concentrate"],
                   default="scatter",
                   help="Початкове розміщення: scatter (випадково) або "
                        "concentrate (всі в центрі — для демо дифузії).")
    p.add_argument("--out-dir", type=str, default="out", help="Каталог артефактів.")
    p.add_argument("--no-animation", action="store_true",
                   help="Не генерувати анімацію (швидше).")
    p.add_argument("--no-plots", action="store_true",
                   help="Не генерувати статичні графіки.")
    p.add_argument("--quiet", action="store_true", help="Менше логу.")

    p.add_argument("--benchmark", action="store_true",
                   help="Прогнати повний набір бенчмарків і вийти.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.benchmark:
        from analysis import run_full_benchmark
        run_full_benchmark(out_dir=out_dir)
        return 0

    cfg = SimulationConfig(
        width=args.width,
        height=args.height,
        n_particles=args.particles,
        p_up=args.p_up,
        p_down=args.p_down,
        p_left=args.p_left,
        p_right=args.p_right,
        total_steps=args.steps,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
        mode=args.mode,
        n_workers=args.workers,
        deadlock_timeout=args.deadlock_timeout,
        initial_layout=args.initial_layout,
        out_dir=out_dir,
        verbose=not args.quiet,
    )

    sim = build_simulation(cfg)
    result = sim.run()

    if not args.no_plots and result.frames:
        suffix = f"{cfg.mode}"
        plot_initial_vs_final(
            result.frames,
            os.path.join(out_dir, f"{suffix}_initial_vs_final.png"),
        )
        plot_total_count(
            result.frames,
            os.path.join(out_dir, f"{suffix}_total_count.png"),
            expected=result.initial_total,
        )
        plot_displacement_variance(
            result.frames,
            os.path.join(out_dir, f"{suffix}_variance.png"),
        )

    if not args.no_animation and result.frames and len(result.frames) > 1:
        try:
            animate_simulation(
                result.frames,
                os.path.join(out_dir, f"{cfg.mode}_animation.mp4"),
                title=f"Броунівський рух — режим {cfg.mode}",
            )
        except Exception as exc:
            print(f"[VIS]    анімацію не згенеровано: {exc}")

    return 0 if result.conserved or cfg.mode in ("unsafe", "deadlock_demo") else 1


if __name__ == "__main__":
    sys.exit(main())
