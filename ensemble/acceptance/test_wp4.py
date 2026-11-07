import subprocess
import sys
from pathlib import Path


def test_hw_module_exists():
    assert Path("hw.py").exists()
    assert Path("bench_train.py").exists()


def test_hw_api():
    import hw
    for name in ("prepare", "pick_device", "describe", "envs_for"):
        assert hasattr(hw, name), f"hw.{name} missing"


def test_hw_pick_device():
    import hw
    dev = hw.pick_device()
    assert isinstance(dev, str)
    assert dev == "cpu" or dev.startswith("cuda:") or dev.startswith("xpu:")


def test_hw_describe():
    import hw
    hw.prepare()
    text = hw.describe()
    assert "устройство" in text


def test_hw_envs_for():
    import hw
    n = hw.envs_for("cpu")
    assert isinstance(n, int)
    assert n >= 0


def test_hw_threads_positive():
    import hw
    assert hw.THREADS >= 1


def test_bench_train_list_runs():
    r = subprocess.run(
        [sys.executable, "bench_train.py", "--list"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, f"bench_train.py --list failed: {r.stderr[-500:]}"


def test_bench_train_help_runs():
    r = subprocess.run(
        [sys.executable, "bench_train.py", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"bench_train.py --help failed: {r.stderr[-500:]}"
    assert "--quick" in r.stdout or "--list" in r.stdout


def test_bench_train_tuned_file_roundtrip():
    """Если tuned/hw_train.json существует — он валидный JSON с полями device, threads."""
    import json
    p = Path("tuned/hw_train.json")
    if not p.exists():
        return  # ещё не замеряли — не ошибка
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("device", "threads"):
        assert key in data, f"tuned/hw_train.json: нет поля {key}"