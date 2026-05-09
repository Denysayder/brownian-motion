"""Сітка кристалу і варіанти переміщення частинок.

Координати зберігаються як `(x, y)`, а в NumPy-сітці індексація лишається
звичною для матриць: `cells[y, x]`.
"""

from __future__ import annotations

import threading
import time
from typing import Tuple

import numpy as np


Coord = Tuple[int, int]  # (x, y)


class Crystal:
    """Сітка `H x W` з лічильником частинок у кожній клітинці."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width і height мають бути > 0")
        self.width = width
        self.height = height
        # int64 з запасом покриває навіть дуже щільні запуски.
        self.cells: np.ndarray = np.zeros((height, width), dtype=np.int64)
        # По одному mutex-у на клітинку: це дешевше, ніж один великий lock.
        self.locks: list[list[threading.Lock]] = [
            [threading.Lock() for _ in range(width)] for _ in range(height)
        ]

    def scatter_particles(self, n_particles: int, rng: np.random.Generator) -> np.ndarray:
        """Розкидати частинки випадково і повернути їхні координати `(x, y)`."""
        if n_particles <= 0:
            raise ValueError("n_particles має бути > 0")
        xs = rng.integers(0, self.width, size=n_particles)
        ys = rng.integers(0, self.height, size=n_particles)
        # np.add.at коректно додає повторні влучання в одну й ту саму клітинку.
        np.add.at(self.cells, (ys, xs), 1)
        return np.stack([xs, ys], axis=1)

    def concentrate_particles(self, n_particles: int) -> np.ndarray:
        """Покласти всі частинки в центральну клітинку."""
        if n_particles <= 0:
            raise ValueError("n_particles має бути > 0")
        cx, cy = self.width // 2, self.height // 2
        self.cells[cy, cx] += n_particles
        xs = np.full(n_particles, cx, dtype=np.int64)
        ys = np.full(n_particles, cy, dtype=np.int64)
        return np.stack([xs, ys], axis=1)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def linear_index(self, x: int, y: int) -> int:
        """Лінійний індекс клітинки для впорядкування локів."""
        return y * self.width + x

    def count_total(self) -> int:
        """Поточна загальна кількість частинок."""
        return int(self.cells.sum())

    def move_unsafe(self, from_xy: Coord, to_xy: Coord) -> None:
        """Наївне переміщення без локів.

        Тут навмисно лишено read-modify-write із паузою між читанням і
        записом. Так race condition добре відтворюється навіть на запусках,
        де короткий `arr[y, x] += 1` майже не встигає перетнутися між потоками.
        """
        fx, fy = from_xy
        tx, ty = to_xy
        if fx == tx and fy == ty:
            return

        src = self.cells[fy, fx]
        dst = self.cells[ty, tx]

        # Пауза робить вікно між читанням і записом достатнім для перемикання.
        new_src = src - 1
        new_dst = dst + 1
        time.sleep(0)

        self.cells[fy, fx] = new_src
        self.cells[ty, tx] = new_dst

    def move_safe_per_cell(self, from_xy: Coord, to_xy: Coord) -> None:
        """Переміщення з двома локами у стабільному порядку."""
        fx, fy = from_xy
        tx, ty = to_xy
        if fx == tx and fy == ty:
            return

        idx_from = self.linear_index(fx, fy)
        idx_to = self.linear_index(tx, ty)

        if idx_from < idx_to:
            first_lock = self.locks[fy][fx]
            second_lock = self.locks[ty][tx]
        else:
            first_lock = self.locks[ty][tx]
            second_lock = self.locks[fy][fx]

        with first_lock:
            with second_lock:
                self.cells[fy, fx] -= 1
                self.cells[ty, tx] += 1

    def move_deadlock_demo(self, from_xy: Coord, to_xy: Coord) -> None:
        """Переміщення з локами у порядку source -> destination.

        Якщо інший потік у цей час робить дзеркальне переміщення, легко
        отримати класичний deadlock:

            T1: lock(A); ... lock(B)   ← чекає B
            T2: lock(B); ... lock(A)   ← чекає A

        Коротка затримка між двома acquire робить сценарій відтворюванішим.
        """
        fx, fy = from_xy
        tx, ty = to_xy
        if fx == tx and fy == ty:
            return

        first_lock = self.locks[fy][fx]
        second_lock = self.locks[ty][tx]

        first_lock.acquire()
        try:
            # Без паузи deadlock залежить від дуже вузького вікна планувальника.
            _busy_wait_us(50)
            second_lock.acquire()
            try:
                self.cells[fy, fx] -= 1
                self.cells[ty, tx] += 1
            finally:
                second_lock.release()
        finally:
            first_lock.release()


def _busy_wait_us(microseconds: int) -> None:
    """Активне очікування приблизно на `microseconds` мікросекунд."""
    import time

    end = time.perf_counter_ns() + microseconds * 1_000
    while time.perf_counter_ns() < end:
        pass
