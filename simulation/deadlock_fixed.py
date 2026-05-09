"""Deadlock-сценарій з виправленням через стабільний порядок locks."""

from __future__ import annotations

import threading
from typing import List

from snapshot import Snapshot, SnapshotCollector

from .base import Simulation


class DeadlockFixedSimulation(Simulation):
    mode_name = "deadlock_fixed"

    def _run_simulation(self) -> tuple[List[Snapshot], bool]:
        cfg = self.config
        crystal = self.crystal
        assert crystal is not None
        particles = self.particles

        n_workers = len(particles)
        collector = SnapshotCollector(
            n_workers=n_workers,
            snapshot_every=cfg.snapshot_every,
            total_steps=cfg.total_steps,
        )
        self.collector = collector

        move_fn = crystal.move_safe_per_cell

        def worker(particle) -> None:
            for step in range(1, cfg.total_steps + 1):
                particle.step(
                    move_fn,
                    width=cfg.width,
                    height=cfg.height,
                    p_up=cfg.p_up,
                    p_down=cfg.p_down,
                    p_left=cfg.p_left,
                    p_right=cfg.p_right,
                )
                collector.worker_sync_if_snapshot_step(step)

        threads = [
            threading.Thread(target=worker, args=(p,), name=f"atom-{p.id}", daemon=True)
            for p in particles
        ]

        if cfg.verbose:
            print(
                f"[RUN]    starting {n_workers} threads in DEADLOCK_FIXED mode "
                "(global lock ordering)"
            )

        started: list[threading.Thread] = []
        try:
            for t in threads:
                t.start()
                started.append(t)
        except RuntimeError as exc:
            if cfg.verbose:
                print(
                    f"[ERROR]  ОС-ліміт потоків: {exc}. "
                    f"Запущено {len(started)}/{n_workers}."
                )
            collector.stop()
            for t in started:
                t.join(timeout=2.0)
            raise

        collector.collector_loop(lambda: crystal.cells)

        for t in threads:
            t.join()

        return collector.frames, False
