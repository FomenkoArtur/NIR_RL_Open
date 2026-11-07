# WP4: bench_train / hw — максимум ресурсов

## Назначение
Обеспечить автоматический выбор устройства (CUDA/XPU/CPU), максимальные потоки,
честные замеры и запись `tuned/hw_train.json`.

## Ссылки на работу
3.1.2.

## Файлы
- `hw.py` — выбор устройства и потоков
- `bench_train.py` — замеры и запись `tuned/hw_train.json`

## Требования
1. `hw.pick_device()` возвращает "cuda:N" при наличии NVIDIA, "xpu:N" для Intel, иначе "cpu".
2. `hw.prepare()` устанавливает OMP/MKL threads, CUDA_DEVICE_ORDER.
3. `bench_train.py` замеряет итерацию PPO при разных (device, threads, envs) и пишет `tuned/hw_train.json`.
4. При отсутствии CUDA — CPU с максимумом потоков.
5. `hw.envs_for(dev_type)` возвращает число параллельных сред.
6. `hw.describe()` возвращает строку с устройством и потоками.
7. `bench_train.py --list` выводит информацию о доступных устройствах без запуска замеров.

## Запреты
- Не менять публичный API `hw.py`: `prepare()`, `pick_device()`, `describe()`, `envs_for()`.
- Не отключать CUDA/XPU детекцию.
- Не удалять поддержку CPU.

## Критерии приёмки
См. test_wp4.py.