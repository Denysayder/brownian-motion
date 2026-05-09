"""Режими симуляції броунівського руху."""

from .base import Simulation, SimulationResult
from .deadlock_demo import DeadlockDemoSimulation
from .deadlock_fixed import DeadlockFixedSimulation
from .pooled import PooledSimulation
from .safe_per_cell import SafePerCellSimulation
from .unsafe import UnsafeSimulation

# Реєстр режимів використовується CLI та бенчмарками.
SIMULATION_MODES: dict[str, type[Simulation]] = {
    "safe_per_cell": SafePerCellSimulation,
    "unsafe": UnsafeSimulation,
    "deadlock_demo": DeadlockDemoSimulation,
    "deadlock_fixed": DeadlockFixedSimulation,
    "pooled": PooledSimulation,
}


def build_simulation(config) -> Simulation:
    """Створити симуляцію для `config.mode`."""
    cls = SIMULATION_MODES.get(config.mode)
    if cls is None:
        raise ValueError(
            f"Невідомий режим: {config.mode!r}. "
            f"Доступні: {sorted(SIMULATION_MODES)}"
        )
    return cls(config)


__all__ = [
    "Simulation",
    "SimulationResult",
    "SafePerCellSimulation",
    "UnsafeSimulation",
    "DeadlockDemoSimulation",
    "DeadlockFixedSimulation",
    "PooledSimulation",
    "SIMULATION_MODES",
    "build_simulation",
]
