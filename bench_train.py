"""
Замер скорости обучения RL-регулятора (PPO) на ЭТОМ компьютере и выбор режима вычислений по ВСЕМУ оборудованию:
видеокарты NVIDIA (CUDA, каждая отдельно), графика Intel (XPU), процессор. Для каждого устройства подбираются:
число потоков процессора (и закрепление потоков, Windows) и число параллельно считаемых сред PPO (на видеокарте
это главный параметр загрузки). Лучший режим записывается в tuned/hw_train.json, оттуда его читает hw.py.

Запуск:  python bench_train.py              (ноутбук - в зарядку, остальные программы закрыть)
         python bench_train.py --quick      короче: 1 прогревочная и 2 замерочные итерации
         python bench_train.py --device cuda   мерить только устройства этого типа или номера (cpu, cuda, cuda:1, xpu)
         python bench_train.py --no-prune   мерить потоки и среды на всех устройствах, а не только на самых быстрых

Как идёт выбор (замер - настоящая итерация PPO: сбор опыта + обновление, как в rl_trainer.train_rl):
  1) каждое устройство - по одному замеру при настройках по умолчанию (все устройства обязательно);
  2) для устройств не медленнее чем в PRUNE раз лучшего - перебор числа потоков процессора и закрепления;
  3) для победителя - перебор числа сред: берётся наибольшее, при котором итерация дольше не более чем в ENV_SLACK раз
     (больше сред = больше опыта за итерацию почти даром, пока видеокарта не загружена); меньше значения по умолчанию - никогда.
Каждый режим - отдельный процесс (потоки и закрепление нельзя поменять после import torch).
"""

import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import hw
from config_rl import RL

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "tuned", "hw_train.json")
PRUNE, ENV_SLACK = 3.0, 1.25
ENV_STEPS = {"cpu": [32, 64, 128, 256], "xpu": [32, 64, 128, 256, 512], "cuda": [128, 256, 512, 1024, 2048, 4096]}
MB_PER_ENV = 0.33 * 1.6                      # память на одну среду PPO (буферы опыта + запас), МБ


# ------------------------------------------------ дочерние процессы ------------------------------------------------
def child_list():
    """Что видит PyTorch (отдельный процесс, чтобы родитель не занимал память видеокарты)."""
    print("INFO " + json.dumps(hw.torch_info()), flush=True)


def child():
    """Один режим: несколько итераций PPO, печатает время итерации."""
    hw.prepare()
    import numpy as np
    import torch
    hw.configure_torch_threads()
    from config import SEED
    from config_rl import RL as RLC, RL_ARCH, RL_TRAIN
    import rl_core as R

    dev = torch.device(hw.pick_device())
    envs = hw.envs_for(dev.type)
    if envs:
        RLC['PPO']['N_ENVS'][dev.type] = envs
    warm, timed = int(os.environ.get("NIR_BENCH_WARM", 2)), int(os.environ.get("NIR_BENCH_TIMED", 3))
    np.random.seed(SEED)
    try:
        agent = R.PPO(dict(RL_ARCH), dict(RL_TRAIN), dev, SEED, log=lambda text: None)
        agent.vf_warmup = 0
        laps = []
        for i in range(warm + timed):                  # прогрев: cudnn.benchmark, запись CUDA Graph
            t = time.perf_counter()
            agent.iterate()                            # внутри есть синхронизация (.item()), время честное
            if i >= warm:
                laps.append(time.perf_counter() - t)
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else 0.0
        result = {"iter_sec": statistics.median(laps), "n_env": agent.n_env, "device": str(dev),
                  "threads": torch.get_num_threads(), "peak_mb": round(peak)}
    except Exception as exc:                           # нехватка памяти, неподдерживаемая операция и т.п.
        result = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"}
    print("RESULT " + json.dumps(result), flush=True)


def _spawn(flag, extra_env, timeout, marker):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **extra_env)
    env.pop("KMP_AFFINITY", None)
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), flag], env=env, cwd=HERE, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "превышено время ожидания"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {"error": tail[-1] if tail else f"код возврата {proc.returncode}"}


def run_child(key, args):
    device, threads, pin, envs = key
    return _spawn("--child", {"NIR_DEVICE": device, "NIR_THREADS": str(threads), "NIR_PIN": pin or "none",
                              "NIR_ENVS": str(envs), "NIR_BENCH_WARM": str(1 if args.quick else 2),
                              "NIR_BENCH_TIMED": str(2 if args.quick else 3)}, 3600, "RESULT ")


# ------------------------------------------------ родительский процесс ------------------------------------------------
def kind_of(device):
    return device.split(":")[0]


def label(key, names):
    device, threads, pin, envs = key
    return f"{device} ({names.get(device, '')}), потоков {threads}, закрепление {pin or 'нет'}, сред {envs}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--device", default=None, help="cpu, cuda, cuda:1, xpu ... (по умолчанию - все)")
    args = ap.parse_args()
    if args.child:
        child()
        return
    if args.list:
        child_list()
        return

    info = _spawn("--list", {}, 300, "INFO ")
    if "error" in info:
        print(f"PyTorch не запустился ({info['error']}); считаем только процессор.")
        info = {"torch": "?", "cuda": None, "hip": None, "devices": [
            {"id": "cpu", "kind": "cpu", "name": hw.cpu_name(), "mem_gb": 0, "sm": 0, "cc": "", "backend": "CPU"}]}
    print(hw.report(info) + "\n")
    devices = [d for d in info["devices"] if args.device is None or d["id"].startswith(args.device)]
    if not devices:
        print("Нет устройств, подходящих под --device.")
        return
    names = {d["id"]: d["name"] for d in info["devices"]}
    mem = {d["id"]: d["mem_gb"] for d in info["devices"]}
    phys = hw.physical_cores()
    runs, meta = {}, {}

    def measure(key):
        result = run_child(key, args)
        if "error" in result:
            print(f"  {label(key, names)}: ошибка ({result['error']})", flush=True)
            return None
        if kind_of(result["device"]) != kind_of(key[0]):
            print(f"  {label(key, names)}: недоступно, расчёт ушёл на {result['device']}", flush=True)
            return None
        runs.setdefault(key, []).append(result["iter_sec"])
        meta[key] = result
        print(f"  {label(key, names)}: {result['iter_sec']:.2f} с на итерацию"
              + (f", пик памяти {result['peak_mb']} МБ" if result["peak_mb"] else ""), flush=True)
        return result

    def best_of(keys):
        keys = [k for k in keys if k in runs]
        return min(keys, key=lambda k: min(runs[k])) if keys else None

    # 1) каждое устройство при настройках по умолчанию
    print("Этап 1: все устройства при настройках по умолчанию")
    base = {}
    for d in devices:
        kind = d["kind"]
        envs = int(RL["PPO"]["N_ENVS"].get(kind, 64))
        threads = phys if kind == "cpu" else max(1, min(4, phys))
        key = (d["id"], threads, "", envs)
        if measure(key):
            base[d["id"]] = key
    if not base:
        print("\nНи один режим не удалось замерить, tuned/hw_train.json не изменён.")
        return
    t_best = min(min(runs[k]) for k in base.values())

    # 2) потоки процессора и закрепление - для устройств, которые могут победить
    print("\nЭтап 2: потоки процессора")
    for dev_id, key in base.items():
        if not args.no_prune and min(runs[key]) > PRUNE * t_best:
            print(f"  {dev_id}: пропущен (медленнее лучшего в {min(runs[key]) / t_best:.1f} раза)")
            continue
        gpu = kind_of(dev_id) != "cpu"
        for threads in hw.thread_candidates(gpu):
            for pin in ([""] if gpu else hw.pin_candidates(threads)):
                cand = (dev_id, threads, pin, key[3])
                if cand not in runs:
                    measure(cand)
        top = sorted((k for k in runs if k[0] == dev_id and k[3] == key[3]), key=lambda k: min(runs[k]))[:2]
        for k in reversed(top):                           # перепроверка двух лучших (ноутбук нагревается)
            measure(k)

    winner = best_of(list(runs))
    dev_id = winner[0]

    # 3) число сред для победителя
    print(f"\nЭтап 3: число параллельных сред на {dev_id}")
    kind = kind_of(dev_id)
    default_envs = winner[3]
    t_default = min(runs[winner])
    limit_mb = mem.get(dev_id, 0) * 1024 * 0.6
    for envs in ENV_STEPS.get(kind, [64]):
        if envs <= default_envs or (kind != "cpu" and limit_mb and envs * MB_PER_ENV > limit_mb):
            continue
        res = measure((dev_id, winner[1], winner[2], envs))
        if res is None or res["iter_sec"] > 3 * t_default:
            break
    same = [k for k in runs if k[:3] == winner[:3]]
    ok = [k for k in same if k[3] >= default_envs and min(runs[k]) <= ENV_SLACK * t_default]
    chosen = max(ok, key=lambda k: k[3])
    t_chosen = min(runs[chosen])

    print("\nИтог (лучшее время итерации PPO):")
    base_key = base.get("cpu")
    for key in sorted(runs, key=lambda k: min(runs[k])):
        speedup = f"{min(runs[base_key]) / min(runs[key]):.1f}x" if base_key in runs else "-"
        print(f"  {min(runs[key]):8.2f} с  {speedup:>6}  {label(key, names)}{'   <- выбран' if key == chosen else ''}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    iters = int(RL["PPO"]["ITERS"])
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"device": chosen[0], "threads": chosen[1], "pin": chosen[2], "envs": chosen[3],
                   "est_iter_sec": round(t_chosen, 2), "name": names.get(chosen[0], ""),
                   "host": platform.node(), "measured": datetime.datetime.now().isoformat(timespec="minutes")},
                  f, indent=2, ensure_ascii=False)
    print(f"\nЗаписано: {OUT_FILE}")
    print(f"Выбрано: {label(chosen, names)}")
    print(f"Оценка: {iters} итераций PPO - примерно {t_chosen * iters / 60:.0f} мин.")


if __name__ == "__main__":
    main()