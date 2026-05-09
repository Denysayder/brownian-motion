"""Спільна основа для режимів симуляції."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import SimulationConfig
from crystal import Crystal
from particle import Particle, build_particles
from snapshot import Snapshot, SnapshotCollector


@dataclass
class SimulationResult:
    """Підсумок одного запуску симуляції."""

    mode: str
    config: SimulationConfig
    initial_total: int
    final_total: int
    frames: List[Snapshot]
    wall_time_s: float
    deadlock_detected: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def diff(self) -> int:
        return self.final_total - self.initial_total

    @property
    def conserved(self) -> bool:
        return self.diff == 0

    def format_report(self) -> str:
        lines = [
            f"[CONFIG] {self.config.describe()}",
            f"[RESULT] Initial: {self.initial_total} | "
            f"Final: {self.final_total} | "
            f"Diff: {self.diff:+d}  "
            f"{'✓' if self.conserved else '✗ (НЕ ЗБЕРІГАЄТЬСЯ — race condition!)'}",
            f"[TIME]   Wall: {self.wall_time_s:.3f}s | "
            f"Snapshots collected: {len(self.frames)}",
        ]
        if self.deadlock_detected:
            lines.append("[DEADLOCK] виявлено зависання потоків (timeout)")
        if self.extra:
            for k, v in self.extra.items():
                lines.append(f"[EXTRA]  {k}: {v}")
        return "\n".join(lines)


class Simulation:
    """Базовий клас для всіх режимів."""

    mode_name: str = "base"

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.crystal: Optional[Crystal] = None
        self.particles: List[Particle] = []
        self.collector: Optional[SnapshotCollector] = None

    def setup(self) -> None:
        """Створити сітку і початковий набір частинок."""
        cfg = self.config
        self.crystal = Crystal(width=cfg.width, height=cfg.height)
        if cfg.initial_layout == "concentrate":
            initial_xy = self.crystal.concentrate_particles(cfg.n_particles)
        else:
            master_rng = np.random.default_rng(cfg.seed)
            initial_xy = self.crystal.scatter_particles(cfg.n_particles, master_rng)
        self.particles = build_particles(initial_xy, master_seed=cfg.seed)

    def run(self) -> SimulationResult:
        """Виконати повний цикл симуляції і повернути SimulationResult."""
        if self.crystal is None:
            self.setup()
        assert self.crystal is not None

        cfg = self.config
        initial_total = self.crystal.count_total()
        if cfg.verbose:
            print(f"[CONFIG] {cfg.describe()}")
            print(f"[INIT]   Initial total particles in crystal: {initial_total}")

        # Початковий кадр потрібен для графіків "до/після".
        initial_snapshot = Snapshot(
            step=0,
            cells=self.crystal.cells.copy(),
            total=initial_total,
        )

        t0 = time.perf_counter()
        frames, deadlock = self._run_simulation()
        wall_time = time.perf_counter() - t0

        # При deadlock collector може не встигнути зібрати жодного кадру.
        all_frames = [initial_snapshot, *frames]
        final_total = self.crystal.count_total()

        result = SimulationResult(
            mode=self.mode_name,
            config=cfg,
            initial_total=initial_total,
            final_total=final_total,
            frames=all_frames,
            wall_time_s=wall_time,
            deadlock_detected=deadlock,
        )
        if cfg.verbose:
            print(result.format_report())
        return result

    def _run_simulation(self) -> tuple[list[Snapshot], bool]:
        """Повернути `(frames, deadlock_detected)`."""
        raise NotImplementedError
