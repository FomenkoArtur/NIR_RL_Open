#!/usr/bin/env python
"""
ensemble/run_ensemble.py
Ансамблирование бесплатных моделей OpenCode: generate -> verify -> judge -> fuse.
Запуск:  python ensemble/run_ensemble.py all --wp WP1 --spec ensemble/specs/WP1.md --test ensemble/acceptance/test_wp1.py
"""
import argparse, json, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".ensemble"
OC = shutil.which("opencode") or "opencode"
SERVER = os.environ.get("OC_URL", "http://127.0.0.1:4096")
MSG = ("Read the file TASK.md in the root of the working directory and carry it out completely. "
       "Edit only files inside this directory.")

PERSONAS = [
    "Сначала продумай решение, запиши краткий план в PLAN.md, затем реализуй.",
    "Сначала внимательно прочитай тесты приёмки и существующий код, затем делай минимальные точные правки.",
    "Сначала проверь физическую и математическую корректность (размерности, знаки, устойчивость), потом пиши код.",
    "Работай маленькими шагами: после каждого шага запускай тесты приёмки и исправляй.",
]

TEMPLATE = """# Роль
Ты — инженер по системам автоматического управления и Python/PyTorch. {persona}

# Спецификация работы
{spec}

# Приёмка
Критерий готовности — прохождение тестов `{test}`. Запускай: `python -m pytest {test} -q --tb=short`.
Папки `ensemble/acceptance` и `ensemble/specs` НЕ ИЗМЕНЯЙ (правки будут откачены).
Не подгоняй код под тесты в обход смысла. Закончив, кратко напиши, что сделано. Не делай git commit.
"""

def sh(cmd, cwd, timeout=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT after {timeout}s: {cmd[:3]}"

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

def ask(model, text, cwd, tag, timeout):
    (cwd / "TASK.md").write_text(text, encoding="utf-8")
    url = os.environ.get("OC_URL", "").strip()
    cmd = [OC, "run"]
    if url:
        cmd += ["--server", url]
    cmd += ["-m", model, "--auto", "--format", "json", "--title", tag, MSG]
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        rc = p.returncode
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, out = 124, f"TIMEOUT after {timeout}s"
    (cwd / f"_oc_{tag}.log").write_text(out[-20000:], encoding="utf-8")
    return rc

def verify(cwd, test, timeout=1800):
    rc, out = sh([sys.executable, "-m", "pytest", test, "-q", "--tb=short", "-p", "no:cacheprovider"],
                 cwd, timeout)
    num = lambda w: int(m.group(1)) if (m := re.search(rf"(\d+) {w}", out)) else 0
    passed, failed, errors = num("passed"), num("failed"), num(r"errors?")
    return dict(ok=(rc == 0 and passed > 0), passed=passed, failed=failed, errors=errors,
                score=passed - failed - errors, tail=out[-3500:])

def commit(cwd, msg):
    git("add", "-A", "--", ".", ":(exclude)TASK.md", ":(exclude)PLAN.md", ":(exclude)_oc_*.log", cwd=cwd)
    git("-c", "user.name=ens", "-c", "user.email=ens@local", "commit", "-q", "--allow-empty", "-m", msg, cwd=cwd)

def candidate(args, base, spec, idx, model):
    name = safe(model)
    wt = make_worktree(args.wp, name, base)
    first = TEMPLATE.format(persona=PERSONAS[idx % len(PERSONAS)], spec=spec, test=args.test)
    task, res = first, {}
    for r in range(args.rounds + 1):
        ask(model, task, wt, f"{args.wp}-{name}-r{r}", args.timeout)
        git("checkout", base, "--", "ensemble/acceptance", "ensemble/specs", cwd=wt)
        res = verify(wt, args.test)
        print(f"[{args.wp}] {name} round {r}: passed={res['passed']} failed={res['failed']}", flush=True)
        if res["ok"]:
            break
        task = first + ("\n\n# Предыдущая попытка не прошла приёмку. Вывод pytest:\n```\n" + res["tail"] +
                        "\n```\nИсправь КОД (не тесты). Текущее состояние кода уже лежит в папке.")
    commit(wt, f"{args.wp}: candidate {name}")
    return dict(name=name, model=model, rounds_used=r, **{k: v for k, v in res.items() if k != "tail"}, tail=res["tail"])

def stage_gen(args, cfg):
    WORK.mkdir(exist_ok=True)
    base, spec = base_branch(), Path(args.spec).read_text(encoding="utf-8")
    models = args.models.split(",") if args.models else cfg["generators"]
    with ThreadPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(candidate, args, base, spec, i, m) for i, m in enumerate(models)]
        cands = []
        for m, f in zip(models, futs):
            try:
                cands.append(f.result())
            except Exception as e:
                print("FAILED", m, e)
    (WORK / args.wp).mkdir(parents=True, exist_ok=True)
    (WORK / args.wp / "results.json").write_text(json.dumps(dict(base=base, cands=cands), ensure_ascii=False, indent=1), "utf-8")
    for c in sorted(cands, key=lambda c: -c["score"]):
        print(f"  {c['name']:35s} ok={c['ok']} score={c['score']} rounds={c['rounds_used']}")

JUDGE = """# Роль
Ты — строгий независимый рецензент кода (системы управления + PyTorch).
# Что сравнить
В папке CANDS/ лежат решения: `<имя>.diff` и `<имя>.log`. Спецификация — в SPEC.md.
Оцени: соответствие спецификации, физическую корректность, отсутствие хаков, читаемость, риски.
# Результат
Запиши ТОЛЬКО JSON в файл `{out}`:
{{"ranking": ["имя_лучшего", "...", "имя_худшего"],
  "best_ideas": [{{"from": "имя", "what": "что перенять", "files": ["path.py"]}}],
  "notes": "кратко"}}
Имена: {names}
"""

FUSE = """# Роль
Ты — ведущий инженер, объединяешь лучшие решения.
# Задача
Текущая папка — лучшее решение-победителя. Рецензенты выделили идеи:
{ideas}
Диффы лежат в CANDS/. Перенеси ТОЛЬКО перечисленные идеи, если они не ломают победителя.
Приёмка: `python -m pytest {test} -q --tb=short` должна остаться зелёной.
Спецификацию см. в SPEC.md. Тесты не менять. Git commit не делать.
"""

def prepare_cands(wt, wp, base, cands, spec):
    d = wt / "CANDS"; d.mkdir(exist_ok=True)
    (wt / "SPEC.md").write_text(spec, encoding="utf-8")
    for c in cands:
        diff = git("diff", f"{base}...ens/{wp}/{c['name']}")[1]
        (d / f"{c['name']}.diff").write_text(diff[:150000], encoding="utf-8")
        (d / f"{c['name']}.log").write_text(f"score={c['score']} ok={c['ok']}\n{c['tail']}", encoding="utf-8")

def stage_judge(args, cfg):
    R = json.loads((WORK / args.wp / "results.json").read_text("utf-8"))
    base, cands = R["base"], R["cands"]
    spec = Path(args.spec).read_text(encoding="utf-8")
    names = [c["name"] for c in cands]
    def one(jm):
        jn = "judge_" + safe(jm)
        wt = make_worktree(args.wp, jn, base)
        prepare_cands(wt, args.wp, base, cands, spec)
        out = f"VERDICT_{safe(jm)}.json"
        ask(jm, JUDGE.format(out=out, names=names), wt, f"{args.wp}-{jn}", args.timeout)
        try:
            return json.loads((wt / out).read_text("utf-8"))
        except Exception as e:
            print("judge failed", jm, e); return None
    with ThreadPoolExecutor(args.parallel) as ex:
        verdicts = [v for v in ex.map(one, cfg["judges"]) if v]
    elig = [c for c in cands if c["ok"]] or cands
    borda = {c["name"]: 0 for c in cands}
    for v in verdicts:
        rk = [n for n in v.get("ranking", []) if n in borda]
        for pos, n in enumerate(rk):
            borda[n] += len(rk) - pos
    win = max(elig, key=lambda c: (borda[c["name"]], c["score"]))
    R.update(verdicts=verdicts, borda=borda, winner=win["name"])
    (WORK / args.wp / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), "utf-8")
    print("borda:", borda, "\nWINNER:", win["name"])

def stage_fuse(args, cfg):
    R = json.loads((WORK / args.wp / "results.json").read_text("utf-8"))
    base, win = R["base"], R["winner"]
    wt = WORK / args.wp / win
    before = verify(wt, args.test)
    ideas = [i for v in R["verdicts"] for i in v.get("best_ideas", []) if i.get("from") != win]
    if not ideas:
        print("нет идей для переноса")
    else:
        spec = Path(args.spec).read_text(encoding="utf-8")
        prepare_cands(wt, args.wp, base, [c for c in R["cands"] if c["name"] != win], spec)
        ask(cfg["fuser"], FUSE.format(ideas=json.dumps(ideas, ensure_ascii=False, indent=1), test=args.test),
            wt, f"{args.wp}-fuse", args.timeout)
        git("checkout", base, "--", "ensemble/acceptance", "ensemble/specs", cwd=wt)
        after = verify(wt, args.test)
        if after["ok"] and after["score"] >= before["score"]:
            commit(wt, f"{args.wp}: fused into {win}")
            print("fusion accepted", before["score"], "->", after["score"])
        else:
            git("reset", "--hard", "HEAD", cwd=wt)
            git("clean", "-fdq", "-e", "TASK.md", cwd=wt)
            print("fusion rejected")
    print(f"\nИтог: ветка ens/{args.wp}/{win}\n"
          f"Посмотри: git diff {base}...ens/{args.wp}/{win}\n"
          f"Принять:   git merge --no-ff ens/{args.wp}/{win}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "judge", "fuse", "all"])
    ap.add_argument("--wp", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--config", default="ensemble/models.json")
    ap.add_argument("--models", help="через запятую")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text("utf-8")) if Path(a.config).exists() else {}
    if a.stage in ("gen", "all"): stage_gen(a, cfg)
    if a.stage in ("judge", "all"): stage_judge(a, cfg)
    if a.stage in ("fuse", "all"): stage_fuse(a, cfg)