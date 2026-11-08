"""
tune_cnn.py - обучение RL-регулятора фиксированной архитектуры 8 -> 8. Подбор архитектуры удалён:
сеть задана в config_rl.RL_ARCH. Optuna перебирает только гиперпараметры обучения PPO (LR, weight decay,
размер батча), а критерий - замкнутый контур (метрика J на выборке рандомизированных объектов).

Запуск:  python tune_cnn.py [испытаний] [воркеров]     по умолчанию 60 и число воркеров из tuned/hw_tune.json
         (hw_tune.py подбирает его под ваше железо; без замера - 2 воркера на каждую видеокарту)
Воркеры раскладываются по всем устройствам одного типа (все видеокарты CUDA / все XPU / процессор). Бюджет испытаний
общий: быстрое устройство делает больше. Прерванный подбор продолжается - испытания лежат в tuned/cnn_rl_study.db.
Результат: tuned/cnn_rl.json (гиперпараметры, отчёт) и tuned/controller_rl.pt (веса); гиперпараметры подхватывает
config_rl.py, полное обучение - python rl_trainer.py или кнопка в GUI.
"""

import hw

hw.prepare()                                                 # до import torch (внутри воркеров - тоже, при spawn)

import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import optuna
import torch

import hw_tune
import rl_core as R
from config import SEED
from config_rl import RL, RL_ARCH

HERE = os.path.dirname(os.path.abspath(__file__))
TUNED = os.path.join(HERE, 'tuned')
STUDY_NAME = 'cnn_rl'
STORAGE_URL = 'sqlite:///' + os.path.join(TUNED, 'cnn_rl_study.db').replace('\\', '/')


# ============================ Optuna ============================
def make_storage():
    # heartbeat: испытания убитых процессов (кнопка «Остановить», закрытое окно) помечаются проваленными, а не висят RUNNING
    return optuna.storages.RDBStorage(STORAGE_URL, engine_kwargs={'connect_args': {'timeout': 120}},
                                      heartbeat_interval=30, grace_period=120)


def make_pruner():
    s = RL['SEARCH']                                         # ресурс Hyperband - итерации PPO короткого прогона
    return optuna.pruners.HyperbandPruner(min_resource=s['EVAL_EVERY'], max_resource=s['PROXY_ITERS'], reduction_factor=3)


def started(study):
    """Сколько испытаний уже взято в работу (завершённые, отсечённые, проваленные, идущие)."""
    return sum(1 for t in study.get_trials(deepcopy=False) if t.state != optuna.trial.TrialState.WAITING)


def wait_enqueued(study, wid, timeout=600):
    """Ручное испытание из очереди берёт только воркер 0 (SQLite не блокирует строку), остальные ждут."""
    for _ in range(timeout):
        if wid == 0 or not any(t.state == optuna.trial.TrialState.WAITING for t in study.get_trials(deepcopy=False)):
            return
        time.sleep(1.0)


MANUAL_HP = {'LR': 3e-4, 'WEIGHT_DECAY': 1e-8, 'BATCH_SIZE': 2048}   # первое испытание - параметры по умолчанию


def suggest_hp(trial):
    """Пространство поиска только гиперпараметров обучения; архитектура фиксирована (8 -> 8)."""
    return {'LR': trial.suggest_float('LR', 1e-4, 3e-3, log=True),
            'WEIGHT_DECAY': trial.suggest_float('WEIGHT_DECAY', 1e-8, 1e-2, log=True),
            'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [1024, 2048, 4096])}


def split_rl(p):
    """Параметры испытания -> (фиксированная архитектура, гиперпараметры обучения)."""
    train = {k: p[k] for k in ('LR', 'WEIGHT_DECAY', 'BATCH_SIZE')}
    return dict(RL_ARCH), train


# ============================ воркер ============================
def worker(wid, target_started):
    """Один воркер: свои устройство и потоки (их задал родитель через NIR_DEVICE / NIR_THREADS), общий бюджет испытаний."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hw.configure_torch_threads()
    dev = torch.device(hw.pick_device())
    tag, s = f"[w{wid} {dev}]", RL['SEARCH']
    eval_ep = R.to_device(R.suite_dr(), dev)                 # одна выборка для всех кандидатов, сразу на устройстве

    def objective(trial):
        p = suggest_hp(trial)
        arch, thp = split_rl(p)
        ok, info, excess = R.arch_limits(arch)               # параметры, размер, время вывода - до обучения
        for k, v in info.items():
            trial.set_user_attr(k, v)
        t_trial = time.time()
        print(f"{tag} trial {trial.number} старт: {p}; {info}", flush=True)
        if not ok:                                           # штраф растёт с превышением - TPE уходит из области
            print(f"{tag} trial {trial.number}: превышены пределы сети ({excess:.2f}), не обучается", flush=True)
            return s['LIMIT_PENALTY'] * (1.0 + excess)
        agents = [R.PPO(arch, thp, dev, SEED + k, log=lambda *_: None) for k in range(s['PROXY_SEEDS'])]
        if RL['IMIT']['ENABLED'] and RL['IMIT']['IN_PROXY']:
            for ag in agents:
                ag.pretrain_imitation()
        cost = float('inf')
        for it in range(1, s['PROXY_ITERS'] + 1):            # короткий прогон RL, seed идут в ногу
            if hw_tune.stop_requested():
                raise optuna.TrialPruned()
            for ag in agents:
                ag.iterate()
            if it % s['EVAL_EVERY'] == 0 or it == s['PROXY_ITERS']:
                cost = float(np.mean([R.rl_evaluate(R.RLPolicy(ag.ac), eval_ep, dev)['J'] for ag in agents]))
                trial.report(cost, it)                       # промежуточное значение для Hyperband
                if trial.should_prune():
                    print(f"{tag} trial {trial.number} ОТСЕЧЁН на итерации {it} (t={time.time() - t_trial:.1f} c)", flush=True)
                    raise optuna.TrialPruned()
        cost = cost if np.isfinite(cost) else 1e3
        print(f"{tag} trial {trial.number} финиш: J={cost:.4f}, {time.time() - t_trial:.1f} c", flush=True)
        return cost

    study = optuna.load_study(study_name=STUDY_NAME, storage=make_storage(),
                              sampler=optuna.samplers.TPESampler(seed=SEED + wid, constant_liar=True),
                              pruner=make_pruner())
    wait_enqueued(study, wid)
    while not hw_tune.stop_requested() and started(study) < target_started:
        study.optimize(objective, n_trials=1, catch=(RuntimeError,))   # нехватка памяти и т.п. - испытание FAIL, воркер живёт


# ============================ финалисты ============================
def _final_task(task):
    """Полное обучение одного кандидата с одним seed на заданном устройстве -> (номер, seed, метрики по выборкам, веса)."""
    number, arch, thp, seed, dev_id = task
    hw.configure_torch_threads()
    dev = torch.device(dev_id)
    t0 = time.time()
    agent = R.PPO(arch, thp, dev, seed, log=lambda *_: None)
    if RL['IMIT']['ENABLED']:
        agent.pretrain_imitation()
    for _ in range(RL['SEARCH']['FULL_ITERS']):
        agent.iterate()
    runs = {k: R.rl_evaluate(R.RLPolicy(agent.ac), ep, dev) for k, ep in R.make_suites(dev).items()}
    print(f"[final] trial {number}, seed {seed}: J(DR) = {runs['dr']['J']:.4f}, {time.time() - t0:.0f} c", flush=True)
    return number, seed, runs, {k: v.cpu() for k, v in agent.ac.state_dict().items()}


def finalize(study, devices, n_workers):
    """Финалисты: полное обучение по нескольким seed (параллельно по устройствам), сравнение с ПИД, сохранение."""
    s = RL['SEARCH']
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value < s['LIMIT_PENALTY']]
    done = sorted(done, key=lambda t: t.value)[:s['FINALISTS']]
    if not done:
        print("[tune_cnn] Нет завершённых испытаний в пределах ограничений", flush=True)
        return
    tasks = []
    for tr in done:
        arch, thp = split_rl(tr.params)
        for seed in s['FINAL_SEEDS']:
            tasks.append((tr.number, arch, thp, seed, devices[len(tasks) % len(devices)]))
    print(f"[tune_cnn] Финалисты: {len(done)} кандидатов × {len(s['FINAL_SEEDS'])} seed, процессов {min(len(tasks), n_workers)}",
          flush=True)
    with mp.get_context('spawn').Pool(min(len(tasks), n_workers)) as pool:
        out = pool.map(_final_task, tasks)

    best = None
    for tr in done:
        mine = [o for o in out if o[0] == tr.number]
        j = float(np.mean([o[2]['dr']['J'] for o in mine]))   # выбор - по среднему J на объектах DR
        if best is None or j < best[0]:
            best = (j, tr, mine)
    j, tr, mine = best
    arch, thp = split_rl(tr.params)
    first = next(o for o in mine if o[1] == s['FINAL_SEEDS'][0])   # веса сохраняются для первого seed (42)

    cpu = torch.device('cpu')
    suites = R.make_suites(cpu)
    results = {'ПИД (эталон)': {k: [R.rl_evaluate(R.PIDBaseline(), ep, cpu)] for k, ep in suites.items()},
               f'RL (trial {tr.number})': {k: [o[2][k] for o in mine] for k in suites}}   # среднее ± разброс по seed
    R.print_report(results)

    os.makedirs(TUNED, exist_ok=True)
    with open(os.path.join(TUNED, 'cnn_rl.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': arch, 'train': thp, 'cost_proxy': tr.value, 'J_dr_final': [o[2]['dr']['J'] for o in mine],
                   'seeds': list(s['FINAL_SEEDS']), 'limits': tr.user_attrs, 'feat': RL['FEAT'],
                   'ref_mode': RL['REF']['MODE'], 'report': results}, f, indent=2, ensure_ascii=False)
    torch.save({'arch': arch, 'feat': RL['FEAT'], 'ref_mode': RL['REF']['MODE'], 'state_dict': first[3]},
               os.path.join(TUNED, 'controller_rl.pt'))
    print("Сохранено в tuned/cnn_rl.json и tuned/controller_rl.pt", flush=True)


# ============================ главная ============================
def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cfg = hw_tune.settings()
    devices = cfg['devices']
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else cfg['workers']
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.makedirs(TUNED, exist_ok=True)
    hw_tune.clear_stop()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED, constant_liar=True),
                                pruner=make_pruner(), storage=make_storage(), study_name=STUDY_NAME, load_if_exists=True)
    existing = started(study)
    print(f"[tune_cnn] Study '{STUDY_NAME}': испытаний уже {len(study.trials)}; новых {n_trials}; воркеров {n_workers} "
          f"на устройствах {devices}, потоков на воркер {cfg['threads']} ({cfg['source']}); "
          f"архитектура фиксирована: {RL_ARCH}; ПИД-эталон: {RL['PID']}", flush=True)
    if len(study.trials) == 0:
        study.enqueue_trial(MANUAL_HP)                       # ручные гиперпараметры - первое испытание

    ctx = mp.get_context('spawn')                            # spawn обязателен при работе с CUDA
    saved = {k: os.environ.get(k) for k in ('NIR_DEVICE', 'NIR_THREADS', 'NIR_PIN')}
    t0, procs = time.time(), []
    for i in range(n_workers):                               # дочерний процесс наследует окружение на момент старта
        os.environ.update(NIR_DEVICE=devices[i % len(devices)], NIR_THREADS=str(cfg['threads']), NIR_PIN='none')
        p = ctx.Process(target=worker, args=(i, existing + n_trials))
        p.start()
        procs.append(p)
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    for p in procs:
        p.join()
    print(f"[tune_cnn] Оптимизация завершена за {time.time() - t0:.1f} c", flush=True)

    if hw_tune.stop_requested():
        print("Остановлено пользователем, финальное дообучение пропущено.", flush=True)
        hw_tune.clear_stop()
        return
    finalize(optuna.load_study(study_name=STUDY_NAME, storage=make_storage()), devices, n_workers)


if __name__ == '__main__':
    main()