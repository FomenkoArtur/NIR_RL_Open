"""
hw.py - подбор вычислительных ресурсов под ЛЮБОЙ компьютер: процессор (потоки, закрепление), видеокарты NVIDIA (CUDA),
встроенная и дискретная графика Intel (XPU), AMD (ROCm-сборка PyTorch, она видна как cuda). Ничего не привязано к конкретной модели.

Режим вычислений берётся в таком порядке (первое найденное):
  1) переменные окружения NIR_DEVICE, NIR_THREADS, NIR_PIN, NIR_ENVS;
  2) tuned/hw_train.json - его пишет bench_train.py после замера на ЭТОМ компьютере (файл с другого компьютера игнорируется);
  3) автоматически: самая мощная видеокарта CUDA, иначе XPU, иначе процессор; потоков = числу физических ядер.

NIR_DEVICE - auto | cpu | cuda | cuda:N | xpu | xpu:N.   NIR_THREADS - потоков PyTorch на процессоре.
NIR_PIN - номера логических процессоров через запятую (только Windows, число номеров = числу потоков), none - не закреплять.
NIR_ENVS - сколько сред PPO считать параллельно (на видеокарте это главный параметр загрузки).
NIR_TF32 - 1 (по умолчанию) разрешить TF32 на видеокартах NVIDIA Ampere и новее, 0 - выключить.

Если PyTorch ещё не установлен, модуль не падает: определяется только процессор (CPU), а в отчёте
появляется причина. Детекция CUDA/XPU при этом не отключается - она работает, как только torch доступен.
"""

import contextlib
import ctypes
import functools
import json
import os
import platform
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUNED_FILE = os.path.join(_HERE, "tuned", "hw_train.json")
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _run(cmd, timeout=8):
    """Вывод внешней команды или пустая строка (команды нет / ошибка / таймаут)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                             errors="replace", creationflags=_NO_WINDOW)
        return out.stdout if out.returncode == 0 else ""
    except Exception:
        return ""


# ------------------------------------------------ процессор ------------------------------------------------
@functools.lru_cache(maxsize=None)
def logical_cores():
    return os.cpu_count() or 1


@functools.lru_cache(maxsize=None)
def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"])
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.physicalcpu"])
    else:
        rows = {r for r in _run(["lscpu", "-p=Core,Socket"]).splitlines() if r and not r.startswith("#")}
        out = str(len(rows)) if rows else ""
    digits = re.findall(r"\d+", out)
    return max(1, int(digits[0])) if digits else logical_cores()


@functools.lru_cache(maxsize=None)
def cpu_name():
    name = ""
    if sys.platform == "win32":
        name = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"]).strip()
    elif sys.platform == "darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return re.sub(r"\s+", " ", name or platform.processor() or platform.machine())


def thread_candidates(gpu=False):
    """Числа потоков для перебора: для процессора - степени двойки, физические и логические ядра; для видеокарты - 1..4+."""
    logical, phys = logical_cores(), physical_cores()
    base = {1, 2, 4, min(8, phys)} if gpu else {1, 2, 4, 8, 16, 32, 64, phys // 2, phys, logical}
    cand = sorted(x for x in base if 1 <= x <= logical)
    return cand[:1] + cand[-7:] if len(cand) > 8 else cand


def pin_candidates(threads):
    """Варианты закрепления (только Windows): по одному потоку на физическое ядро, если включён Hyper-Threading."""
    if sys.platform != "win32" or logical_cores() <= physical_cores() or not 2 <= threads <= physical_cores():
        return [""]
    return ["", ",".join(str(2 * i) for i in range(threads))]


# ------------------------------------------------ видеоадаптеры ------------------------------------------------
def _vendor(name):
    n = name.lower()
    for key, vendor in (("nvidia", "nvidia"), ("geforce", "nvidia"), ("quadro", "nvidia"), ("intel", "intel"),
                        ("amd", "amd"), ("radeon", "amd"), ("advanced micro", "amd")):
        if key in n:
            return vendor
    return "other"


@functools.lru_cache(maxsize=None)
def system_gpus():
    """Все видеоадаптеры, которые видит ОС (без PyTorch): [{'name','vendor','mem_gb'}]."""
    found = []
    for line in _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            found.append({"name": parts[0], "vendor": "nvidia",
                          "mem_gb": round(float(re.sub(r"[^\d.]", "", parts[1]) or 0) / 1024, 1)})
    rows = []
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object Name | ConvertTo-Json -Compress"])
        try:
            data = json.loads(out) if out.strip() else []
            rows = [r.get("Name", "") for r in (data if isinstance(data, list) else [data])]
        except ValueError:
            rows = []
    elif sys.platform.startswith("linux"):
        rows = [re.sub(r"^.*?:\s*", "", r.split(": ", 1)[-1]) for r in _run(["lspci"]).splitlines()
                if re.search(r"vga|3d|display", r, re.I)]
    skip = ("microsoft basic", "remote", "virtual", "parsec", "indirect")
    for name in rows:
        vendor = _vendor(name)
        if not name or any(s in name.lower() for s in skip) or (vendor == "nvidia" and found):
            continue
        found.append({"name": name.strip(), "vendor": vendor, "mem_gb": 0.0})
    return found


def _cpu_device():
    """Описание процессора в том же формате, что и устройства PyTorch."""
    return {"id": "cpu", "kind": "cpu", "name": cpu_name(), "mem_gb": 0.0, "sm": 0, "cc": "", "backend": "CPU"}


def torch_info():
    """Что видит PyTorch: версия, сборка CUDA/XPU и устройства (cpu всегда последним).

    Если PyTorch не установлен или не импортируется, возвращается только CPU и ключ torch_error:
    остальной код (отчёты, выбор устройства) продолжает работать.
    """
    try:
        import torch
    except Exception as exc:                                      # torch ещё не установлен - это не ошибка запуска
        return {"torch": None, "cuda": None, "hip": None, "devices": [_cpu_device()],
                "torch_error": f"{type(exc).__name__}: {exc}"}
    devices = []
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devices.append({"id": f"cuda:{i}", "kind": "cuda", "name": p.name,
                                "mem_gb": round(p.total_memory / 2 ** 30, 1),
                                "sm": int(p.multi_processor_count), "cc": f"{p.major}.{p.minor}",
                                "backend": "ROCm" if getattr(torch.version, "hip", None) else "CUDA"})
    except Exception:
        pass                                                      # драйвер/сборка CUDA недоступны - остаются другие устройства
    xpu = getattr(torch, "xpu", None)
    try:
        if xpu is not None and xpu.is_available():
            for i in range(xpu.device_count()):
                p = xpu.get_device_properties(i)
                devices.append({"id": f"xpu:{i}", "kind": "xpu", "name": str(getattr(p, "name", xpu.get_device_name(i))),
                                "mem_gb": round(getattr(p, "total_memory", 0) / 2 ** 30, 1),
                                "sm": int(getattr(p, "gpu_eu_count", 0) or 0), "cc": "", "backend": "XPU"})
    except Exception:
        pass
    devices.append(_cpu_device())
    return {"torch": torch.__version__, "cuda": getattr(torch.version, "cuda", None),
            "hip": getattr(torch.version, "hip", None), "devices": devices}


def _rank(d):
    """Грубая оценка мощности устройства, когда замера ещё не было: число мультипроцессоров, затем память."""
    return (d.get("sm") or 0) * 1000 + (d.get("mem_gb") or 0)


def report(info=None):
    """Текст: всё найденное оборудование и что из него доступно PyTorch (с подсказками, как включить остальное)."""
    info = info or torch_info()
    lines = [f"Процессор: {cpu_name()}; физических ядер {physical_cores()}, логических {logical_cores()}"]
    usable = [d for d in info["devices"] if d["kind"] != "cpu"]
    if info.get("torch"):
        lines.append(f"PyTorch {info['torch']}" + (f", CUDA {info['cuda']}" if info.get("cuda") else
                                                  f", ROCm {info['hip']}" if info.get("hip") else ", сборка без CUDA"))
    else:
        lines.append("PyTorch не установлен: доступен только процессор"
                     + (f" ({info['torch_error']})" if info.get("torch_error") else ""))
    gpus = system_gpus()
    lines.append("Видеоадаптеры в системе: " + ("; ".join(f"{g['name']}" + (f" ({g['mem_gb']} ГБ)" if g["mem_gb"] else "")
                                                         for g in gpus) if gpus else "не найдены"))
    for d in usable:
        lines.append(f"  доступно для обучения: {d['id']} - {d['name']}, {d['backend']}, {d['mem_gb']} ГБ"
                     + (f", {d['sm']} мультипроцессоров, capability {d['cc']}" if d["kind"] == "cuda" else ""))
    cpu = next((d for d in info["devices"] if d["kind"] == "cpu"), None)
    lines.append(f"  доступно для обучения: cpu - {(cpu or {}).get('name') or cpu_name()}, CPU")
    kinds = {d["kind"] for d in usable}
    vendors = {g["vendor"] for g in gpus}
    if "nvidia" in vendors and "cuda" not in kinds:
        lines.append("  ! Видеокарта NVIDIA есть, но PyTorch её не видит: поставьте сборку PyTorch с CUDA "
                     "(https://pytorch.org/get-started/locally/) и свежий драйвер NVIDIA.")
    if "intel" in vendors and "xpu" not in kinds:
        lines.append("  ! Графика Intel есть, но PyTorch её не использует: нужна сборка PyTorch с поддержкой XPU "
                     "(https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).")
    if "amd" in vendors and "cuda" not in kinds:
        lines.append("  ! Видеокарта AMD используется PyTorch только в сборке ROCm.")
    if _tuned_note:
        lines.append("  ! " + _tuned_note)
    return "\n".join(lines)


# ------------------------------------------------ настройки ------------------------------------------------
def _load_tuned():
    try:
        with open(_TUNED_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, ""
    if not isinstance(data, dict):
        return {}, "tuned/hw_train.json имеет неверный формат и игнорируется; запустите bench_train.py"
    if data.get("host") != platform.node():
        return {}, (f"tuned/hw_train.json получен на другом компьютере ({data.get('host') or 'не указан'}) и "
                    f"игнорируется; запустите bench_train.py на этом компьютере")
    return data, ""


_tuned, _tuned_note = _load_tuned()


def _setting(env_name, key, default):
    value = os.environ.get(env_name)
    return value if value is not None and value != "" else _tuned.get(key, default)


try:
    THREADS = max(1, int(_setting("NIR_THREADS", "threads", physical_cores())))
except (TypeError, ValueError):
    THREADS = physical_cores()

PIN = str(os.environ.get("NIR_PIN", _tuned.get("pin", "")) or "")
if PIN.strip().lower() == "none":
    PIN = ""
DEVICE = str(_setting("NIR_DEVICE", "device", "auto")).lower()
TF32 = os.environ.get("NIR_TF32", "1") != "0"


def envs_for(dev_type):
    """Сколько сред PPO считать параллельно на устройстве этого типа; 0 - оставить значение из config_rl."""
    value = os.environ.get("NIR_ENVS", "").strip()
    if value.isdigit():
        return max(0, int(value))
    kind = str(dev_type or "").split(":")[0].strip().lower()        # принимаем и 'cuda', и 'cuda:0'
    tuned_type = str(_tuned.get("device", "")).split(":")[0].strip().lower()
    try:
        tuned_envs = max(0, int(_tuned.get("envs", 0) or 0))
    except (TypeError, ValueError):
        tuned_envs = 0
    return tuned_envs if kind and tuned_type == kind else 0


# ------------------------------------------------ процесс ------------------------------------------------
_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_PROCESS_POWER_THROTTLING = 4
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_prepared = False


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def _configure_windows_process():
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(handle, _ABOVE_NORMAL_PRIORITY_CLASS)
        state = _PowerThrottlingState(1, 1, 0)       # отключить EcoQoS (режим энергосбережения процесса)
        kernel32.SetProcessInformation(handle, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass


def pin_list():
    ids = [p.strip() for p in PIN.split(",") if p.strip()]
    if not ids or not all(p.isdigit() for p in ids) or len(ids) != THREADS:
        return []
    return ids


def prepare():
    """Вызывается до import torch: потоки BLAS/OpenMP, закрепление, порядок видеокарт, режим процесса Windows."""
    global _prepared
    if _prepared:
        return
    _prepared = True
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")      # номера cuda:N как в nvidia-smi
    ids = pin_list()
    if ids and sys.platform == "win32" and "KMP_AFFINITY" not in os.environ:
        os.environ["KMP_AFFINITY"] = "granularity=fine,proclist=[%s],explicit" % ",".join(ids)
    _configure_windows_process()


def configure_torch_threads():
    """Вызывается в каждом рабочем потоке с расчётами PyTorch: число потоков и режимы ускорения видеокарты."""
    import torch
    torch.set_num_threads(THREADS)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if TF32:                                                  # быстрее на Ampere и новее, для RL точности хватает
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")


@functools.lru_cache(maxsize=None)
def _info():
    return torch_info()


@functools.lru_cache(maxsize=None)
def pick_device():
    """Строка устройства для обучения: 'cpu', 'cuda:N' или 'xpu:N'."""
    if DEVICE == "cpu":
        return "cpu"
    gpus = [d for d in _info()["devices"] if d["kind"] != "cpu"]
    if DEVICE != "auto":
        if DEVICE in {d["id"] for d in gpus}:
            return DEVICE
        kind = DEVICE.split(":")[0]
        if ":" not in DEVICE:                                     # указан тип без номера - берём самое мощное
            same = [d for d in gpus if d["kind"] == kind]
            if same:
                return max(same, key=_rank)["id"]
        return "cpu"                                              # запрошенного устройства нет - честный откат на CPU
    for kind in ("cuda", "xpu"):                                  # CUDA в приоритете, затем графика Intel
        same = [d for d in gpus if d["kind"] == kind]
        if same:
            return max(same, key=_rank)["id"]
    return "cpu"


def describe():
    dev = pick_device()
    info = next((d for d in _info()["devices"] if d["id"] == dev), None)
    text = f"устройство {dev}" + (f" ({info['name']}, {info['mem_gb']} ГБ)" if info and dev != "cpu" else f" ({cpu_name()})")
    text += f", потоков процессора: {THREADS}"
    ids = pin_list()
    if ids:
        text += f", закреплены за логическими процессорами {','.join(ids)}"
    envs = envs_for(dev.split(":")[0])
    return text + (f", параллельных сред PPO: {envs}" if envs else "")


def on_battery():
    if sys.platform != "win32":
        return False
    try:
        status = _SystemPowerStatus()
        if ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(status)):
            return status.ACLineStatus == 0
    except Exception:
        pass
    return False


@contextlib.contextmanager
def keep_awake():
    """Пока идёт обучение, Windows не уходит в сон. Действует для потока, который вызвал функцию."""
    set_state = None
    if sys.platform == "win32":
        try:
            set_state = ctypes.WinDLL("kernel32").SetThreadExecutionState
            set_state.argtypes = [ctypes.c_uint32]
            set_state.restype = ctypes.c_uint32
            set_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
        except Exception:
            set_state = None
    try:
        yield
    finally:
        if set_state is not None:
            try:
                set_state(_ES_CONTINUOUS)
            except Exception:
                pass