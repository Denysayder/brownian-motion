"""Параметри одного запуску моделі броунівського руху."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Допустимі режими роботи симулятора.
Mode = Literal[
    "safe_per_cell",
    "unsafe",
    "deadlock_demo",
    "deadlock_fixed",
    "pooled",
]

# Частинка на межі відбивається: фактично лишається в поточній клітинці.
BoundaryPolicy = Literal["reflect"]

# Як початково розмістити частинки:
#   "scatter"     — випадково по сітці;
#   "concentrate" — усі в центральній клітинці.
InitialLayout = Literal["scatter", "concentrate"]


@dataclass
class SimulationConfig:
    """Налаштування симуляції з базовою валідацією значень."""

    # --- Геометрія кристалу ---
    width: int = 50
    height: int = 50

    # --- Атоми домішки ---
    n_particles: int = 500

    # --- Імовірності напрямків ---
    # Сума ймовірностей може бути < 1 — решта = "залишитися на місці".
    p_up: float = 0.25
    p_down: float = 0.25
    p_left: float = 0.25
    p_right: float = 0.25

    # --- Час симуляції та дискретизація знімків ---
    total_steps: int = 1000
    snapshot_every: int = 50

    # --- Випадковість ---
    # Master-seed; дочірні RNG для частинок отримуємо через np.random.SeedSequence.
    seed: int = 42

    # --- Поведінка на межі ---
    boundary_policy: BoundaryPolicy = "reflect"

    # --- Початкове розміщення ---
    initial_layout: InitialLayout = "scatter"

    # --- Режим роботи ---
    mode: Mode = "safe_per_cell"

    # --- Параметри pooled-режиму ---
    n_workers: int = 8

    # --- Поведінка демо-режимів ---
    # Тайм-аут (с), після якого deadlock_demo вважається завислим.
    deadlock_timeout: float = 15.0

    # --- Артефакти ---
    out_dir: str = "out"

    # --- Логування ---
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width і height мають бути > 0")
        if self.n_particles <= 0:
            raise ValueError("n_particles має бути > 0")
        if self.total_steps <= 0:
            raise ValueError("total_steps має бути > 0")
        if self.snapshot_every <= 0:
            raise ValueError("snapshot_every має бути > 0")
        if self.n_workers <= 0:
            raise ValueError("n_workers має бути > 0")
        probs = (self.p_up, self.p_down, self.p_left, self.p_right)
        if any(p < 0 for p in probs):
            raise ValueError("імовірності мають бути ≥ 0")
        if sum(probs) - 1.0 > 1e-9:
            raise ValueError(
                f"Сума ймовірностей переходу > 1: {sum(probs)} "
                "(допустимо < 1 — решта = stay)"
            )

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    @property
    def density(self) -> float:
        return self.n_particles / self.n_cells

    def describe(self) -> str:
        """Короткий опис для шапки логу."""
        return (
            f"mode={self.mode}, grid={self.height}x{self.width}, "
            f"n_particles={self.n_particles}, steps={self.total_steps}, "
            f"snapshot_every={self.snapshot_every}, "
            f"p=(↑{self.p_up} ↓{self.p_down} ←{self.p_left} →{self.p_right}), "
            f"seed={self.seed}, workers={self.n_workers}"
        )
