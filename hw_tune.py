"""
hw_tune.py - параллелизм подбора на ЭТОМ компьютере: сколько воркеров Optuna и на каких устройствах (tune_cnn.py),
сколько процессов считает популяцию CMA-ES (tune_pid.py). Устройства и потоки процессора определяет hw.py.

Замер:  python hw_tune.py            (после bench_train.py; остальные программы закрыть)
        python hw_tune.py --quick    короче: окно замера 10 с вместо 20 с
Результат - tuned/hw_tune.json (только для этого компьютера). settings() читают tune_cnn.py, tune_pid.py и GUI;
без файла берутся разумные значения по умолчанию.

Замер идёт так: нужное число процессов стартует вместе, прогревается, по сигналу родителя все одновременно работают
фиксированное окно, считается суммарная скорость (итераций PPO в секунду / кандидатов ПИД в секунду). Выбирается наименьшее
число процессов, дающее почти максимум (меньше процессов - меньше нагрев и память).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import hw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "tuned", "hw_tune.json")
STOP_FILE = os.path.join(HERE, "tuned", ".stop_tune")      # GUI создаёт файл - подбор мягко останавливается
WINDOW_SEC, PID_POP, TOLERANCE = 20.0, 24, 1.05            # окно замера, кандидатов ПИД на замер, допуск (5 %)


def stop_requested():
    return os.path.exists(STOP_FILE)


def clear_stop():
    try:
        os.remove(STOP_FILE)
    except OSError:
        pass


# ------------------------------------------------ настройки для tune_cnn.py / tune_pid.py ------------------------------------------------
def _load():
    try:
        with open(OUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if data.get("host") == platform.node() else {}      # файл с другого компьютера не используется


def _default_devices():
    dev = hw.pick_device()
    kind = dev.split(":")[0]
    if kind == "cpu" or hw.DEVICE not in ("auto", "cuda", "xpu"):   # явно указанное устройство - только оно
        return [dev]
    return [d["id"] for d in hw._info()["devices"] if d["kind"] == kind]


def pid_procs():
    """Процессов расчёта популяции CMA-ES (без обращения к PyTorch)."""
    return int(_load().get("pid", {}).get("procs") or hw.physical_cores())


def settings():
    """{'devices', 'workers_per_device', 'threads', 'workers', 'pid_procs', 'source'}."""
    data = _load()
    cnn = data.get("cnn", {})
    devices = cnn.get("devices") or _default_devices()
    gpu, phys = devices[0] != "cpu", hw.physical_cores()
    wpd = int(cnn.get("workers_per_device") or (2 if gpu else min(4, phys)))
    threads = int(cnn.get("threads") or (max(1, min(2, phys // (wpd * len(devices)))) if gpu else max(1, phys // wpd)))
    return {"devices": devices, "workers_per_device": wpd, "threads": threads, "workers": wpd * len(devices),
            "pid_procs": pid_procs(), "source": "hw_tune.json" if data else "авто"}


# ------------------------------------------------ дочерние процессы замера ------------------------------------------------
def _sync(args):
    """Прогрев окончен: сообщить родителю и ждать общий старт, чтобы окна замера всех процессов совпали."""
    open(os.path.join(args.sync, f"ready{args.id}"), "w").close()
    while not os.path.exists(os.path.join(args.sync, "go")):
        time.sleep(0.05)


def child_ppo(args):
    hw.prepare()
    import numpy as np
    import torch
    hw.configure_torch_threads()
    from config import SEED
    from config_rl import RL_ARCH, RL_TRAIN
    import rl_core as R

    np.random.seed(SEED)
    try:
        agent = R.PPO(dict(RL_ARCH), dict(RL_TRAIN), torch.device(hw.pick_device()), SEED, log=lambda text: None)
        agent.vf_warmup = 0
        agent.iterate()                                              # прогрев (CUDA Graph, cudnn.benchmark)
        _sync(args)
        t0, n = time.perf_counter(), 0
        while time.perf_counter() - t0 < args.window:
            agent.iterate()
            n += 1
        result = {"iters": n, "sec": time.perf_counter() - t0}
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"}
    print("RESULT " + json.dumps(result), flush=True)


def child_pid(args):
    import numpy as np
    import tune_pid as T
    cases, quiet = T.build_cases()
    rng = np.random.RandomState(args.id)
    xs = list(np.clip(T.encode(*T.MANUAL) + 0.1 * rng.randn(args.chunk, 4), 0, 1))
    T.evaluate(xs[:1], cases, quiet)                                 # прогрев
    _sync(args)
    t0 = time.perf_counter()
    T.evaluate(xs, cases, quiet)
    print("RESULT " + json.dumps({"sec": time.perf_counter() - t0}), flush=True)


def run_group(flag, envs, extra, timeout=900):
    """Запустить len(envs) процессов одновременно -> список результатов (или {'error'})."""
    tmp = tempfile.mkdtemp(prefix="nir_tune_")
    procs = []
    for i, env in enumerate(envs):
        cmd = [sys.executable, os.path.abspath(__file__), flag, "--sync", tmp, "--id", str(i)] + extra
        procs.append(subprocess.Popen(cmd, cwd=HERE, env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **env),
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                      errors="replace"))
    t0 = time.time()
    while time.time() - t0 < timeout and not all(os.path.exists(os.path.join(tmp, f"ready{i}")) for i in range(len(procs))):
        if any(p.poll() is not None for p in procs):                 # кто-то упал до готовности
            break
        time.sleep(0.2)
    open(os.path.join(tmp, "go"), "w").close()
    results = []
    for p in procs:
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
        line = next((x for x in reversed(out.splitlines()) if x.startswith("RESULT ")), None)
        tail = (err or out).strip().splitlines()
        results.append(json.loads(line[7:]) if line else {"error": tail[-1] if tail else "нет ответа"})
    return results


# ------------------------------------------------ родитель: замер и выбор ------------------------------------------------
def pick(rates):
    """rates: {число процессов: скорость} -> наименьшее число процессов с почти максимальной скоростью."""
    best = max(rates.values())
    return min(k for k, v in rates.items() if v * TOLERANCE >= best)


def measure_cnn(info, window):
    kinds = {d["kind"] for d in info["devices"]}
    kind = str(hw._tuned.get("device", "")).split(":")[0]            # тип устройства, выбранный bench_train.py
    kind = kind if kind in kinds else next((k for k in ("cuda", "xpu") if k in kinds), "cpu")
    devices = [d["id"] for d in info["devices"] if d["kind"] == kind]
    gpu, phys = kind != "cpu", hw.physical_cores()
    cands = [1, 2, 3, 4, 6] if gpu else [w for w in (1, 2, 3, 4, 6, 8, 12, 16) if w <= phys]
    print(f"tune_cnn: устройства {devices}, замер {window:.0f} с на вариант")
    rates, threads_of = {}, {}
    for w in cands:
        total = w * len(devices)
        threads = max(1, min(2, phys // total)) if gpu else max(1, phys // total)
        envs = [{"NIR_DEVICE": devices[i % len(devices)], "NIR_THREADS": str(threads), "NIR_PIN": "none"}
                for i in range(total)]
        res = run_group("--ppo", envs, ["--window", str(window)])
        bad = [r["error"] for r in res if "error" in r]
        if bad:
            print(f"  воркеров на устройство {w}: ошибка ({bad[0]})", flush=True)
            break
        rate = sum(r["iters"] / r["sec"] for r in res)
        rates[w], threads_of[w] = rate, threads
        print(f"  воркеров на устройство {w} (потоков {threads}): {rate:.2f} итераций PPO/с суммарно", flush=True)
        if rate < 0.9 * max(rates.values()):                         # пик пройден
            break
    if not rates:
        return None
    w = pick(rates)
    return {"devices": devices, "workers_per_device": w, "threads": threads_of[w], "iters_per_sec": round(rates[w], 3)}


def measure_pid(window):
    logical = hw.logical_cores()
    cands = sorted({p for p in (1, 2, 4, 8, 16, 32, hw.physical_cores(), logical) if 1 <= p <= logical})
    print("tune_pid: процессов расчёта популяции CMA-ES")
    rates = {}
    for p in cands:
        chunk = max(1, PID_POP // p)
        env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        res = run_group("--pid", [env] * p, ["--chunk", str(chunk)])
        if any("error" in r for r in res):
            print(f"  процессов {p}: ошибка ({next(r['error'] for r in res if 'error' in r)})", flush=True)
            break
        rates[p] = chunk * p / max(r["sec"] for r in res)
        print(f"  процессов {p}: {rates[p]:.2f} кандидатов/с", flush=True)
    if not rates:
        return None
    p = pick(rates)
    return {"procs": p, "cand_per_sec": round(rates[p], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo", action="store_true")
    ap.add_argument("--pid", action="store_true")
    ap.add_argument("--sync", default="")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=1)
    ap.add_argument("--window", type=float, default=WINDOW_SEC)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.ppo:
        child_ppo(args)
        return
    if args.pid:
        child_pid(args)
        return

    import bench_train
    info = bench_train._spawn("--list", {}, 300, "INFO ")
    if "error" in info:
        print(f"PyTorch не запустился ({info['error']}); считаем только процессор.")
        info = {"devices": [{"id": "cpu", "kind": "cpu"}]}
    window = 10.0 if args.quick else WINDOW_SEC
    cnn, pid = measure_cnn(info, window), measure_pid(window)
    out = {"host": platform.node(), "measured": datetime.datetime.now().isoformat(timespec="minutes")}
    if cnn:
        out["cnn"] = cnn
    if pid:
        out["pid"] = pid
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nЗаписано: {OUT_FILE}")
    if cnn:
        print(f"tune_cnn: {cnn['workers_per_device']} воркера(ов) на каждое из устройств {cnn['devices']}, "
              f"потоков на воркер {cnn['threads']}")
    if pid:
        print(f"tune_pid: {pid['procs']} процесс(ов)")


if __name__ == "__main__":
    main()