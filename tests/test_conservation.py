"""Тести збереження кількості частинок."""

from __future__ import annotations

from config import SimulationConfig
from simulation import build_simulation


def _run(mode: str, **overrides) -> tuple[int, int]:
    cfg = SimulationConfig(
        width=overrides.pop("width", 30),
        height=overrides.pop("height", 30),
        n_particles=overrides.pop("n_particles", 200),
        total_steps=overrides.pop("total_steps", 200),
        snapshot_every=overrides.pop("snapshot_every", 200),
        seed=overrides.pop("seed", 42),
        mode=mode,
        n_workers=overrides.pop("n_workers", 4),
        verbose=False,
        **overrides,
    )
    sim = build_simulation(cfg)
    res = sim.run()
    return res.initial_total, res.final_total


def test_safe_per_cell_conserves_particles():
    initial, final = _run("safe_per_cell")
    assert initial == final, (
        f"safe_per_cell зламав інваріант: initial={initial}, final={final}"
    )


def test_pooled_conserves_particles():
    initial, final = _run("pooled")
    assert initial == final, (
        f"pooled зламав інваріант: initial={initial}, final={final}"
    )


def test_pooled_scales_to_large_n():
    """Pooled має витримувати N, незручне для підходу "1 потік на атом"."""
    initial, final = _run(
        "pooled",
        width=100, height=100,
        n_particles=10_000,
        total_steps=50,
        snapshot_every=50,
        n_workers=8,
    )
    assert initial == final == 10_000


def test_deadlock_fixed_conserves_particles():
    initial, final = _run(
        "deadlock_fixed",
        width=5, height=5, n_particles=40,
        total_steps=200, snapshot_every=200,
    )
    assert initial == final


def test_concentrate_layout_starts_at_center():
    """`concentrate` кладе всі частинки в одну центральну клітинку."""
    cfg = SimulationConfig(
        width=20, height=20, n_particles=100,
        total_steps=1, snapshot_every=1, seed=42, mode="pooled",
        n_workers=2, initial_layout="concentrate", verbose=False,
    )
    sim = build_simulation(cfg)
    sim.setup()
    assert sim.crystal is not None
    cx, cy = 20 // 2, 20 // 2
    assert sim.crystal.cells[cy, cx] == 100
    assert int(sim.crystal.cells.sum()) == 100


def test_unsafe_breaks_invariant_under_high_contention():
    """Race condition має відтворюватися при високій контенції."""
    for params in [
        dict(width=6, height=6, n_particles=200, total_steps=200),
        dict(width=8, height=8, n_particles=400, total_steps=400),
        dict(width=5, height=5, n_particles=800, total_steps=400),
    ]:
        initial, final = _run("unsafe", snapshot_every=10**9, **params)
        if final != initial:
            return  # OK, race спостерігається
    raise AssertionError(
        f"Race condition не відтворився жодного разу. "
        f"Останній запуск: initial={initial}, final={final}. "
        "Перевір налаштування move_unsafe (має лишатися time.sleep(0))."
    )
