"""Deadlock-сценарій: два locks беруться без спільного порядку."""

from __future__ import annotations

import threading
import time
from typing import List

from snapshot import Snapshot, SnapshotCollector

from .base import Simulation


class DeadlockDemoSimulation(Simulation):
    mode_name = "deadlock_demo"

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

        move_fn = crystal.move_deadlock_demo

        def worker(particle) -> None:
            try:
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
            except Exception:
                return

        threads = [
            threading.Thread(target=worker, args=(p,), name=f"atom-{p.id}", daemon=True)
            for p in particles
        ]

        if cfg.verbose:
            print(
                f"[RUN]    starting {n_workers} threads in DEADLOCK_DEMO mode "
                f"(timeout {cfg.deadlock_timeout}s)"
            )

        for t in threads:
            t.start()

        # Колектор працює в головному потоці, watchdog зупиняє його при зависанні.
        watchdog = _DeadlockWatchdog(threads, collector, cfg.deadlock_timeout)
        watchdog.start()
        try:
            collector.collector_loop(lambda: crystal.cells)
        except Exception:
            pass

        deadline = time.monotonic() + 2.0
        for t in threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)
        watchdog.cancel()

        deadlocked = any(t.is_alive() for t in threads) or watchdog.fired
        if deadlocked and cfg.verbose:
            n_alive = sum(t.is_alive() for t in threads)
            print(
                f"[DEADLOCK DETECTED] {n_alive}/{n_workers} потоків лишилися "
                "заблокованими на захопленні локів. Це і є очікувана поведінка."
            )
        elif cfg.verbose:
            print(
                "[NOTE] Deadlock у цьому запуску НЕ відтворився "
                "(іноді буває — спробуйте збільшити particles/steps або зменшити сітку)."
            )

        return collector.frames, deadlocked


class _DeadlockWatchdog(threading.Thread):
    """Таймер, який перериває колектор після `timeout` секунд."""

    def __init__(
        self,
        worker_threads: list[threading.Thread],
        collector: SnapshotCollector,
        timeout: float,
    ) -> None:
        super().__init__(name="deadlock-watchdog", daemon=True)
        self.worker_threads = worker_threads
        self.collector = collector
        self.timeout = timeout
        self._cancelled = threading.Event()
        self.fired = False

    def run(self) -> None:
        end = time.monotonic() + self.timeout
        while time.monotonic() < end:
            if self._cancelled.is_set():
                return
            if not any(t.is_alive() for t in self.worker_threads):
                return
            time.sleep(0.1)
        if self._cancelled.is_set():
            return
        self.fired = True
        self.collector.stop()

    def cancel(self) -> None:
        self._cancelled.set()
