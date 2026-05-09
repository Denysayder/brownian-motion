# Brownian motion

Невелика симуляція броунівського руху в 2D-кристалі.

У сітці є частинки домішки. На кожному кроці вони випадково рухаються вгору,
вниз, ліворуч або праворуч. Після запуску перевіряється головний інваріант:
скільки частинок було на початку, стільки ж має лишитися в кінці.

У проекті є кілька режимів:

- `safe_per_cell` — базовий безпечний варіант, один потік на частинку.
- `unsafe` — спеціально без синхронізації, щоб показати race condition.
- `deadlock_demo` / `deadlock_fixed` — приклад deadlock і виправлення.
- `pooled` — масштабований варіант через `ThreadPoolExecutor`.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python main.py --mode safe_per_cell \
    --width 50 --height 50 --particles 500 \
    --steps 1000 --snapshot-every 50
```

Результати, графіки й анімації зберігаються в `out/`.

## Корисні команди

```bash
# тести
python -m pytest tests/ -v

# повний набір бенчмарків і графіків
python main.py --benchmark

# великий запуск без анімації та графіків
python main.py --mode pooled --particles 100000 --steps 100 \
    --workers 8 --no-animation --no-plots
```

## Основні файли

- `main.py` — CLI.
- `config.py` — параметри симуляції.
- `crystal.py` — сітка і переміщення частинок.
- `particle.py` — логіка одного атома домішки.
- `snapshot.py` — знімки стану сітки.
- `simulation/` — різні режими запуску.
- `analysis.py` — бенчмарки та графіки.
