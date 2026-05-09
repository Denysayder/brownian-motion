"""Знімки стану кристалу.

У режимах з потоками колектор синхронізується з воркерами через `Barrier`:
сітка копіюється тільки тоді, коли ніхто її не змінює. У pooled-режимі бар'єр
не потрібен, бо `executor.map(...)` уже дочекався завершення кроку.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class Snapshot:
    """Копія сітки на конкретному кроці симуляції."""

    step: int
    cells: np.ndarray
    total: int


@dataclass
class SnapshotCollector:
    """Колектор знімків з опційною бар'єрною синхронізацією."""

    n_workers: int
    snapshot_every: int
    total_steps: int
    frames: List[Snapshot] = field(default_factory=list)
    barrier: threading.Barrier = field(init=False)
    _stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        if self.n_workers <= 0:
            raise ValueError("n_workers має бути > 0")
        if self.snapshot_every <= 0:
            raise ValueError("snapshot_every має бути > 0")
        # +1 для потоку-колектора.
        self.barrier = threading.Barrier(self.n_workers + 1)

    def worker_sync_if_snapshot_step(self, step: int) -> None:
        """Синхронізувати воркер, якщо на цьому кроці потрібен знімок."""
        if self._is_snapshot_step(step):
            self._safe_wait()  # 1) очікуємо, поки колектор зніме знімок
            self._safe_wait()  # 2) колектор сигналізує "продовжуйте"

    def collector_loop(self, get_cells_view) -> None:
        """Збирати бар'єрні знімки протягом симуляції."""
        for step in range(1, self.total_steps + 1):
            if self._stop.is_set():
                break
            if self._is_snapshot_step(step):
                self._safe_wait()  # 1) усі потоки заморожені
                if not self._stop.is_set():
                    self._record_snapshot(step, get_cells_view())
                self._safe_wait()  # 2) відпускаємо потоки

    def stop(self) -> None:
        """Перервати бар'єр, якщо симуляцію треба зупинити достроково."""
        self._stop.set()
        self.barrier.abort()

    def take_snapshot_now(self, step: int, cells: np.ndarray) -> None:
        """Зробити знімок без бар'єра, коли сітка вже не змінюється."""
        if self._is_snapshot_step(step):
            self._record_snapshot(step, cells)

    def _is_snapshot_step(self, step: int) -> bool:
        return (step % self.snapshot_every == 0) or (step == self.total_steps)

    def _record_snapshot(self, step: int, cells: np.ndarray) -> None:
        frame = cells.copy()  # immutable snapshot
        total = int(frame.sum())
        self.frames.append(Snapshot(step=step, cells=frame, total=total))

    def _safe_wait(self) -> None:
        """`barrier.wait()` без шуму при штатному перериванні."""
        if self._stop.is_set():
            return
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            self._stop.set()
