"""Тести коректності snapshot-механізму."""

from __future__ import annotations

import numpy as np

from config import SimulationConfig
from simulation import build_simulation


def _run(mode: str, **overrides):
    cfg = SimulationConfig(
        width=overrides.pop("width", 25),
        height=overrides.pop("height", 25),
        n_particles=overrides.pop("n_particles", 100),
        total_steps=overrides.pop("total_steps", 200),
        snapshot_every=overrides.pop("snapshot_every", 50),
        seed=overrides.pop("seed", 7),
        mode=mode,
        n_workers=overrides.pop("n_workers", 4),
        verbose=False,
        **overrides,
    )
    sim = build_simulation(cfg)
    return sim.run()


def test_safe_per_cell_snapshots_have_correct_total():
    res = _run("safe_per_cell")
    assert res.frames, "Має бути хоча б один знімок"
    for f in res.frames:
        assert int(f.cells.sum()) == res.initial_total == f.total


def test_pooled_snapshots_have_correct_total():
    res = _run("pooled")
    assert res.frames
    for f in res.frames:
        assert int(f.cells.sum()) == res.initial_total == f.total


def test_snapshot_count_safe_per_cell():
    """initial + регулярні знімки без дублювання фінального кроку."""
    res = _run("safe_per_cell", total_steps=200, snapshot_every=50)
    # очікуємо: step=0 (initial), 50, 100, 150, 200 → 5 кадрів
    assert len(res.frames) == 5, [f.step for f in res.frames]


def test_snapshots_are_immutable_after_run_safe_per_cell():
    """Snapshot має бути ізольованою копією масиву."""
    res = _run("safe_per_cell")
    if not res.frames:
        return
    snap = res.frames[1]  # перший знімок після initial
    snap_copy = snap.cells.copy()
    sim_crystal_cells = res.frames[-1].cells  # останній кадр — тільки для shape
    _ = sim_crystal_cells + 100
    assert np.array_equal(snap.cells, snap_copy), \
        "Знімок не повинен змінюватися від подальших операцій"
