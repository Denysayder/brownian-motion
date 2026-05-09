"""Частинка домішки: позиція, ідентифікатор і власний RNG."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


Coord = Tuple[int, int]
MoveFn = Callable[[Coord, Coord], None]  # сигнатура `crystal.move_*`


class Particle:
    """Один атом домішки у сітці кристалу."""

    __slots__ = ("id", "x", "y", "rng")

    def __init__(self, particle_id: int, x: int, y: int, rng: np.random.Generator) -> None:
        self.id = particle_id
        self.x = x
        self.y = y
        self.rng = rng

    def step(
        self,
        move_fn: MoveFn,
        width: int,
        height: int,
        p_up: float,
        p_down: float,
        p_left: float,
        p_right: float,
    ) -> None:
        """Зробити один випадковий крок і оновити позицію частинки."""
        u = float(self.rng.random())

        new_x, new_y = self.x, self.y
        if u < p_up:
            new_y -= 1
        elif u < p_up + p_down:
            new_y += 1
        elif u < p_up + p_down + p_left:
            new_x -= 1
        elif u < p_up + p_down + p_left + p_right:
            new_x += 1
        # else: stay

        # Відбиття від межі: вихід за сітку перетворюється на stay.
        if not (0 <= new_x < width and 0 <= new_y < height):
            new_x, new_y = self.x, self.y

        if new_x == self.x and new_y == self.y:
            return  # no-op, навіть лок брати не треба

        move_fn((self.x, self.y), (new_x, new_y))
        self.x = new_x
        self.y = new_y


def build_particles(
    initial_xy: np.ndarray,
    master_seed: int,
) -> list[Particle]:
    """Створити частинки з незалежними, але відтворюваними RNG."""
    n = len(initial_xy)
    seed_seq = np.random.SeedSequence(master_seed)
    child_seeds = seed_seq.spawn(n)
    particles: list[Particle] = []
    for i, (xy, child_seed) in enumerate(zip(initial_xy, child_seeds)):
        rng = np.random.default_rng(child_seed)
        particles.append(Particle(i, int(xy[0]), int(xy[1]), rng))
    return particles
