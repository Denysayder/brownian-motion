"""Масштабований режим: фіксований пул потоків і chunks частинок."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from particle import Particle
from snapshot import Snapshot, SnapshotCollector

from .base import Simulation


def _chunkify(items: Sequence, n_chunks: int) -> list[Sequence]:
    """Розбити `items` на приблизно рівні шматки."""
    n = len(items)
    if n == 0 or n_chunks <= 0:
        return []
    n_chunks = min(n_chunks, n)
    base, rem = divmod(n, n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


class PooledSimulation(Simulation):
    mode_name = "pooled"

    def _run_simulation(self) -> tuple[List[Snapshot], bool]:
        cfg = self.config
        crystal = self.crystal
        assert crystal is not None
        particles = self.particles

        n_workers = cfg.n_workers
        # У pooled-режимі колектор використовується без бар'єра.
        collector = SnapshotCollector(
            n_workers=n_workers,
            snapshot_every=cfg.snapshot_every,
            total_steps=cfg.total_steps,
        )
        self.collector = collector

        move_fn = crystal.move_safe_per_cell
        chunks = _chunkify(particles, n_workers)

        # Менше lookup-ів у гарячому циклі.
        width = cfg.width
        height = cfg.height
        p_up, p_down, p_left, p_right = cfg.p_up, cfg.p_down, cfg.p_left, cfg.p_right

        def process_chunk(chunk: Sequence[Particle]) -> None:
            for particle in chunk:
                particle.step(
                    move_fn,
                    width=width,
                    height=height,
                    p_up=p_up,
                    p_down=p_down,
                    p_left=p_left,
                    p_right=p_right,
                )

        if cfg.verbose:
            print(
                f"[RUN]    pooled: {n_workers} workers, "
                f"{len(chunks)} chunks, ~{len(particles)//max(1,len(chunks))} atoms/chunk"
            )

        progress_every = max(1, cfg.total_steps // 20)
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="pool") as executor:
            for step in range(1, cfg.total_steps + 1):
                # list(...) змушує дочекатися всіх chunks на поточному кроці.
                list(executor.map(process_chunk, chunks))

                collector.take_snapshot_now(step, crystal.cells)

                if cfg.verbose and (step % progress_every == 0 or step == cfg.total_steps):
                    print(f"[RUN]    step {step}/{cfg.total_steps}")

        return collector.frames, False
