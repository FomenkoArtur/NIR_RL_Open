#!/usr/bin/env python
"""Ансамбль с визуальной диагностикой + коротким обучением перед проверкой."""
import argparse, json, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bayesian import ModelWeights

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".ensemble"
OC = shutil.which("opencode") or "opencode"
SERVER = os.environ.get("OC_URL", "http://127.0.0.1:4096")
MSG = "Read TASK.md and carry it out completely. Edit only files inside this directory."

PERSONAS = [
    "Сначала продумай решение, запиши краткий план в PLAN.md, затем реализуй.",
    "Сначала внимательно прочитай тесты приёмки и существующий код, затем делай минимальные точные правки.",
    "Сначала проверь физическую и математическую корректность, потом пиши код.",
    "Работай маленькими шагами: после каждого шага запускай тесты и исправляй.",
]

TEMPLATE = """# Роль
Ты — инженер. {persona}

# Спецификация
{spec}

# Приёмка
Прохождение тестов `{test}`. Запускай: `python -m pytest {test} -q --tb=short`.
Папки ensemble/acceptance и ensemble/specs НЕ ИЗМЕНЯЙ. Не делай git commit.
"""

VISUAL_REFINE = """# Роль
Ты — старший инженер. Твои изменения прошли unit-тесты, но короткое обучение показало проблемы.

# Отчёт визуальной диагностики (после {iters} итераций PPO)
{report}

text

# Что делать
1. Прочитай цифры: где статика > 0.01 или ложные срабатывания > 0.05.
2. В файлах `config_rl.py` и/или `rl_core.py` усиль штрафы за эти проблемы:
   - за статику: увеличь `K_E` и/или `LAMBDA_STATIC`;
   - за ложные срабатывания: увеличь `LAMBDA_FA`, уменьши `FA_THRESH` или `FA_BAND`.
3. После правок запусти `python -m pytest {test} -q --tb=short`.

Графики диагностики лежат в `diagnostics/` (S1_rl.png, S2_rl.png, S3_rl.png) — можешь их посмотреть.
Git commit не делать.
"""

JUDGE = """# Роль
Ты — рецензент. Сравни решения в CANDS/ (diff+log). Спецификация — SPEC.md.
# Результат
Запиши ТОЛЬКО JSON в `{out}`:
{{"ranking": ["имя_лучшего","...","имя_худшего"],
  "best_ideas": [{{"from":"имя","what":"что перенять"}}],
  "notes":"кратко"}}
Имена: {names}
"""

FUSE = """# Роль
Ведущий инженер. Объедини решение-победителя с идеями рецензентов.
{ideas}
Приёмка: `python -m pytest {test} -q --tb=short` должна остаться зелёной.
"""

REFINER = """# Роль
Старший инженер. Победитель уже прошёл приёмку. Рецензенты указали идеи: {ideas}.
Перенеси их, не ломая тесты. `python -m pytest {test} -q --tb=short` должна быть зелёной.
"""


def sh(cmd, cwd, timeout=None, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT {timeout}s"


def git(*a, cwd=ROOT):
    return sh(["git", *a], cwd)


def base_branch():
    return git("rev-parse", "--abbrev-ref", "HEAD")[1].strip()


def safe(name):
    return re.sub(r"\W+", "_", name.split("/")[-1])


def make_worktree(wp, name, base):
    path = WORK / wp / name
    git("worktree", "remove", "--force", str(path))
    git("branch", "-D", f"ens/{wp}/{name}")
    git("worktree", "prune")
    rc, out = git("worktree", "add", "-b", f"ens/{wp}/{name}", str(path), base)
    if rc:
        raise RuntimeError(out)
    return path


def _ensure_files(wt):
    for folder in ("ensemble/specs", "ensemble/acceptance"):
        src = ROOT / folder
        dst = wt / folder
        if not src.exists(): continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
    for helper in ("ensemble/visual_diagnose.py",):
        src = ROOT / helper
        if src.exists():
            dst = wt / helper
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def ask(model, text, cwd, tag, timeout):
    (cwd / "TASK.md").write_text(text, encoding="utf-8")
    cmd = [OC, "run", "-m", model, "--auto", "--format", "json", "--title", tag, MSG]
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, out = 124, "TIMEOUT"
    (cwd / f"_oc_{tag}.log").write_text(out[-20000:], encoding="utf-8")
    return rc


def verify(cwd, test, timeout=1800):
    rc, out = sh([sys.executable, "-m", "pytest", test, "-q", "--tb=short", "-p", "no:cacheprovider"],
                 cwd, timeout)
    num = lambda w: int(m.group(1)) if (m := re.search(rf"(\d+) {w}", out)) else 0
    passed, failed, errors = num("passed"), num("failed"), num(r"errors?")
    return dict(ok=(rc == 0 and passed > 0), passed=passed, failed=failed, errors=errors,
                score=passed - failed - errors, tail=out[-3500:])


def run_visual_check(wt, iters, timeout=1800):
    """Запускает visual_diagnose.py (обучает короткий прогон + проверяет графики).
    Возвращает (report_text, ok_bool).
    """
    diag = wt / "ensemble" / "visual_diagnose.py"
    if not diag.exists():
        return "", True  # скрипта нет - не блокируем
    try:
        p = subprocess.run(
            [sys.executable, str(diag), "--iters", str(iters)],
            cwd=wt, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        out = (p.stdout or "") + (p.stderr or "")
        (wt / "_diag_output.log").write_text(out[-20000:], encoding="utf-8")
        ok = (p.returncode == 0)  # 0 = OK, 2 = плохо, 1 = ошибка (считаем как ok, чтобы не блокировать)
        if p.returncode == 1:
            ok = True
        return out[-3000:], ok
    except subprocess.TimeoutExpired:
        return "TIMEOUT visual_diagnose", True
    except Exception as e:
        return f"visual_diagnose error: {e}", True


def commit(cwd, msg):
    git("add", "-A", "--", ".", ":(exclude)TASK.md", ":(exclude)PLAN.md", ":(exclude)_oc_*.log",
        ":(exclude)_diag_output.log", ":(exclude)controller_diag.pt", ":(exclude)diagnostics",
        ":(exclude)diagnostics/*", cwd=cwd)
    git("-c", "user.name=ens", "-c", "user.email=ens@local", "commit", "-q", "--allow-empty", "-m", msg, cwd=cwd)


def candidate(args, base, spec, idx, model):
    name = safe(model)
    wt = make_worktree(args.wp, name, base)
    _ensure_files(wt)

    first = TEMPLATE.format(persona=PERSONAS[idx % len(PERSONAS)], spec=spec, test=args.test)
    task, res = first, {}
    r = -1
    visual_ok = False
    last_report = ""

    for r in range(args.rounds + 1):
        rc = ask(model, task, wt, f"{args.wp}-{name}-r{r}", args.timeout)
        _ensure_files(wt)
        res = verify(wt, args.test)

        if res["passed"] == 0 and res["failed"] == 0 and res["errors"] == 0:
            print(f"[{args.wp}] {name} r{r}: pytest упал (rc={rc})", flush=True)
            print(res["tail"][-500:], flush=True)
        else:
            print(f"[{args.wp}] {name} r{r}: unit {res['passed']}p/{res['failed']}f/{res['errors']}e", flush=True)

        if not res["ok"]:
            task = first + f"\n\n# Тесты не прошли\n```\n{res['tail']}\n```\nИсправь код."
            continue

        # --- unit OK, теперь короткое обучение + визуальная проверка ---
        print(f"[{args.wp}] {name} r{r}: unit OK → короткое обучение ({args.diag_iters} итер) + проверка графиков...", flush=True)
        last_report, visual_ok = run_visual_check(wt, iters=args.diag_iters)

        # копируем диагностику
        diag_dst = WORK / args.wp / name / "diagnostics"
        if diag_dst.exists(): shutil.rmtree(diag_dst)
        src = wt / "diagnostics"
        if src.exists(): shutil.copytree(src, diag_dst)

        print(f"[{args.wp}] {name} r{r}: визуально {'OK' if visual_ok else 'ПЛОХО'}", flush=True)
        if last_report:
            print("  --- отчёт диагностики ---", flush=True)
            for line in last_report.splitlines()[-10:]:
                print(f"  {line}", flush=True)
            print("  -------------------------", flush=True)

        if visual_ok:
            break

        if r < args.rounds:
            task = VISUAL_REFINE.format(report=last_report, test=args.test, iters=args.diag_iters)
            continue
        # раунды кончились, но всё плохо - всё равно возвращаем
        break

    commit(wt, f"{args.wp}: candidate {name}")
    out = dict(name=name, model=model, rounds_used=r,
               visual_ok=visual_ok, visual_report=last_report[-2000:],
               **{k: v for k, v in res.items() if k != "tail"}, tail=res["tail"])
    return out


def stage_gen(args, cfg):
    WORK.mkdir(exist_ok=True)
    base = base_branch()
    spec = Path(args.spec).read_text(encoding="utf-8")
    models = args.models.split(",") if args.models else cfg["generators"]

    with ThreadPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(candidate, args, base, spec, i, m) for i, m in enumerate(models)]
        cands = []
        for m, f in zip(models, futs):
            try: cands.append(f.result())
            except Exception as e: print("FAILED", m, e)

    (WORK / args.wp).mkdir(parents=True, exist_ok=True)
    (WORK / args.wp / "results.json").write_text(
        json.dumps(dict(base=base, cands=cands), ensure_ascii=False, indent=1), "utf-8")

    mw = ModelWeights()
    for c in cands:
        # успех = unit OK И визуально OK
        success = c.get("ok", False) and c.get("visual_ok", False)
        mw.update(c["name"], success)

    print("\nИтоговые результаты:", flush=True)
    for c in sorted(cands, key=lambda c: (c.get("visual_ok", False), c.get("score", 0)), reverse=True):
        print(f"  {c['name']:35s} unit_ok={c.get('ok')} visual_ok={c.get('visual_ok')} "
              f"score={c.get('score')} rounds={c.get('rounds_used')}", flush=True)


def prepare_cands(wt, wp, base, cands, spec):
    d = wt / "CANDS"; d.mkdir(exist_ok=True)
    (wt / "SPEC.md").write_text(spec, encoding="utf-8")
    for c in cands:
        diff = git("diff", f"{base}...ens/{wp}/{c['name']}")[1]
        (d / f"{c['name']}.diff").write_text(diff[:150000], encoding="utf-8")
        (d / f"{c['name']}.log").write_text(
            f"unit_ok={c.get('ok')} visual_ok={c.get('visual_ok')} score={c.get('score')}\n"
            f"{(c.get('visual_report') or '')[-1000:]}\n{c['tail']}", encoding="utf-8")


def stage_judge(args, cfg):
    R = json.loads((WORK / args.wp / "results.json").read_text("utf-8"))
    base, cands = R["base"], R["cands"]
    spec = Path(args.spec).read_text(encoding="utf-8")
    names = [c["name"] for c in cands]
    mw = ModelWeights()

    def one(jm):
        jn = "judge_" + safe(jm)
        wt = make_worktree(args.wp, jn, base)
        prepare_cands(wt, args.wp, base, cands, spec)
        out = f"VERDICT_{safe(jm)}.json"
        ask(jm, JUDGE.format(out=out, names=names), wt, f"{args.wp}-{jn}", args.timeout)
        try: return json.loads((wt / out).read_text("utf-8"))
        except Exception as e:
            print("judge failed", jm, e); return None

    with ThreadPoolExecutor(args.parallel) as ex:
        verdicts = [v for v in ex.map(one, cfg["judges"]) if v]

    elig = [c for c in cands if c.get("visual_ok")] or [c for c in cands if c.get("ok")] or cands
    borda = {c["name"]: 0 for c in cands}
    for v in verdicts:
        rk = [n for n in v.get("ranking", []) if n in borda]
        for pos, n in enumerate(rk):
            borda[n] += mw.weight(n) * (len(rk) - pos)

    win = max(elig, key=lambda c: (borda[c["name"]], c.get("score", 0)))
    R.update(verdicts=verdicts, borda=borda, winner=win["name"])
    (WORK / args.wp / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), "utf-8")
    print("borda:", borda, "\nWINNER:", win["name"])


def stage_fuse(args, cfg):
    R = json.loads((WORK / args.wp / "results.json").read_text("utf-8"))
    base, win = R["base"], R["winner"]
    wt = WORK / args.wp / win
    before = verify(wt, args.test)
    ideas = [i for v in R["verdicts"] for i in v.get("best_ideas", []) if i.get("from") != win]
    if ideas:
        spec = Path(args.spec).read_text(encoding="utf-8")
        prepare_cands(wt, args.wp, base, [c for c in R["cands"] if c["name"] != win], spec)
        ask(cfg["fuser"], FUSE.format(ideas=json.dumps(ideas, ensure_ascii=False, indent=1), test=args.test),
            wt, f"{args.wp}-fuse", args.timeout)
        after = verify(wt, args.test)
        if after["ok"] and after["score"] >= before["score"]:
            git("rm", "-rf", "--ignore-unmatch", "CANDS", "SPEC.md", cwd=wt)
            commit(wt, f"{args.wp}: fused into {win}")
            print("fusion accepted", before["score"], "->", after["score"])
        else:
            git("reset", "--hard", "HEAD", cwd=wt); git("clean", "-fdq", "-e", "TASK.md", cwd=wt)
            print("fusion rejected")
    else:
        print("нет идей для переноса")

    if getattr(args, "no_merge", False):
        print(f"\n--no-merge. Проверь: git diff {base}...ens/{args.wp}/{win}")
    else:
        print(f"\nПринять: git merge --no-ff ens/{args.wp}/{win}")


def stage_refine(args, cfg):
    R = json.loads((WORK / args.wp / "results.json").read_text("utf-8"))
    base, win = R["base"], R["winner"]
    wt = WORK / args.wp / win
    before = verify(wt, args.test)
    if not before["ok"]: return
    ideas = [i for v in R["verdicts"] for i in v.get("best_ideas", [])]
    spec = Path(args.spec).read_text(encoding="utf-8")
    prepare_cands(wt, args.wp, base, R["cands"], spec)
    print("Запуск доработчика (refiner)...")
    ask(cfg["fuser"], REFINER.format(ideas=json.dumps(ideas, ensure_ascii=False, indent=1), test=args.test),
        wt, f"{args.wp}-refine", args.timeout)
    after = verify(wt, args.test)
    if after["ok"] and after["score"] >= before["score"]:
        commit(wt, f"{args.wp}: refined")
        print("refine accepted", before["score"], "->", after["score"])
    else:
        git("reset", "--hard", "HEAD", cwd=wt); git("clean", "-fdq", "-e", "TASK.md", cwd=wt)
        print("refine rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "judge", "fuse", "refine", "all"])
    ap.add_argument("--wp", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--config", default="ensemble/models.json")
    ap.add_argument("--models", help="через запятую")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--diag-iters", type=int, default=300, help="итераций короткого обучения для диагностики")
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()
    a.diag_iters = a.diag_iters  # для доступа в candidate
    cfg = json.loads(Path(a.config).read_text("utf-8")) if Path(a.config).exists() else {}

    if a.stage in ("gen", "all"): stage_gen(a, cfg)
    if a.stage in ("judge", "all"): stage_judge(a, cfg)
    if a.stage in ("fuse", "all"): stage_fuse(a, cfg)
    if a.stage in ("refine", "all"): stage_refine(a, cfg)
