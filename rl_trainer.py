"""
rl_trainer.py - обучение RL-регулятора (PPO) и сравнительный отчёт. Единственная реализация обучения: её вызывают
кнопка «Обучить регулятор» (gui/workers.py), main.py и этот файл из командной строки.
Режим вычислений (устройство, потоки, число сред) - из hw.py / tuned/hw_train.json (его делает bench_train.py).

Запуск:  python rl_trainer.py                    обучить (seed 42), сохранить controller_rl.pt, отчёт
         python rl_trainer.py --seeds 42,43,44   обучить по каждому seed, в отчёте среднее ± разброс
         python rl_trainer.py --eval             только отчёт по controller_rl.pt (или tuned/controller_rl.pt из Colab)
         python rl_trainer.py --iters 100        другое число итераций PPO
"""

import hw

hw.prepare()                                                    # до import torch: потоки, закрепление, режим процесса

import argparse
import os
import time

import numpy as np
import torch

from config import SEED
from config_rl import RL, RL_ARCH, RL_TRAIN
import rl_core as R

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, 'controller_rl.pt')               # веса RL-регулятора
COLAB_FILE = os.path.join(HERE, 'tuned', 'controller_rl.pt')    # веса, полученные в Colab (tune_cnn.py)


def get_device():
    return torch.device(hw.pick_device())                       # cpu / cuda:N / xpu:N по hw_train.json и NIR_DEVICE


def train_rl(arch=None, train_hp=None, seed=SEED, iters=None, log=print, on_iter=None, should_stop=None,
             validate=True):
    """
    Имитация ПИД (если включена) + PPO. Возвращает ActorCritic на CPU в режиме eval или None, если остановлено до 1-й итерации.
    on_iter(it, stats) - после каждой итерации (stats: ret, kl и val_J, если в эту итерацию считалась проверка);
    should_stop() - опрашивается перед итерацией; validate - раз в EVAL_EVERY итераций оценивать регулятор на фиксированной
    выборке рандомизированных объектов (метрика J, меньше - лучше) и в конце вернуть лучшие по J веса.
    """
    arch, train_hp = dict(arch or RL_ARCH), dict(train_hp or RL_TRAIN)
    iters = RL['PPO']['ITERS'] if iters is None else int(iters)  # --iters 0: только имитация
    dev = get_device()
    suite = R.to_device(R.suite_val(), dev) if validate else None  # номинал + DR с шумом до x1.5
    envs = hw.envs_for(dev.type)                                # число сред под это железо (bench_train.py)
    if envs:
        RL['PPO']['N_ENVS'][dev.type] = envs
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    agent = R.PPO(arch, train_hp, dev, seed, log=log)
    log(f"RL: {hw.describe()}; сред {agent.n_env}, архитектура {arch}")
    suite = R.to_device(R.suite_dr(), dev) if validate else None
    if RL['IMIT']['ENABLED']:
        log("Предобучение имитацией ПИД-эталона...")
        agent.pretrain_imitation()
    every = int(RL['SEARCH']['EVAL_EVERY'])
    best_j, best_state, done, t0 = float('inf'), None, 0, time.perf_counter()
    if suite is not None and RL['IMIT']['ENABLED']:             # веса сразу после имитации - тоже кандидат: PPO не должен их ухудшить
        best_j = R.rl_evaluate(R.RLPolicy(agent.ac, RL['FEAT']), suite, dev)['J']
        best_state = {k: v.detach().clone() for k, v in agent.ac.state_dict().items()}
        log(f"После имитации: J = {best_j:.3f}")
    for it in range(1, iters + 1):
        if should_stop is not None and should_stop():
            log("Обучение остановлено пользователем.")
            break
        st = agent.iterate()
        done = it
        if suite is not None and (it == 1 or it % every == 0 or it == iters):
            j = R.rl_evaluate(R.RLPolicy(agent.ac, RL['FEAT']), suite, dev)['J']
            st['val_J'] = j
            if j < best_j:
                best_j, best_state = j, {k: v.detach().clone() for k, v in agent.ac.state_dict().items()}
        if on_iter is not None:
            on_iter(it, st)
        if it == 1 or it % 10 == 0:
            log(f"  итерация {it}: доход {st['ret']:.3f}, KL {st['kl']:.4f}, "
                f"{(time.perf_counter() - t0) / it:.1f} с/итерация")
    if done == 0 and best_state is None:
        return None
    if best_state is not None:
        agent.ac.load_state_dict(best_state)
        log(f"Взяты лучшие веса по проверочной метрике: J = {best_j:.3f}")
    return agent.ac.cpu().eval()


def save(ac, path=None):
    """Сохранить веса вместе с описанием входа/выхода (загрузка не зависит от текущего config_rl)."""
    path = str(path or OUT_FILE)
    torch.save({'arch': ac.hp, 'feat': RL['FEAT'], 'ref_mode': RL['REF']['MODE'],
                'state_dict': ac.state_dict()}, path)
    return path


def load(path=None):
    """Загрузить RL-регулятор: controller_rl.pt, иначе tuned/controller_rl.pt из Colab. -> (feat, ActorCritic)."""
    path = str(path or (OUT_FILE if os.path.exists(OUT_FILE) else COLAB_FILE))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет весов RL-регулятора: ни {OUT_FILE}, ни {COLAB_FILE}. "
                                f"Сначала обучите регулятор (кнопка в GUI или python rl_trainer.py)")
    ck = torch.load(path, map_location='cpu', weights_only=False)                   # обучено на GPU - грузится на CPU
    if ck.get('ref_mode', RL['REF']['MODE']) != RL['REF']['MODE']:
        raise ValueError(f"Веса обучены с эталоном '{ck['ref_mode']}', а в config_rl сейчас '{RL['REF']['MODE']}' "
                         f"(переменная NIR_RL_REF)")
    ac = R.ActorCritic(ck['arch'])
    ac.load_state_dict(ck['state_dict'])
    print(f"Загружено: {path}")
    return ck.get('feat', RL['FEAT']), ac.eval()


def report(models, log=print):
    """models: [(feat, ActorCritic)] по seed. Таблицы: ПИД-эталон и RL на всех оценочных выборках."""
    dev = get_device()
    suites = R.make_suites(dev)                                 # номинал, DR, вне диапазонов, C1-C9
    results = {'ПИД (эталон)': {k: [R.rl_evaluate(R.PIDBaseline(), ep, dev)] for k, ep in suites.items()}}
    results['RL'] = {k: [R.rl_evaluate(R.RLPolicy(ac.to(dev), feat), ep, dev) for feat, ac in models]
                     for k, ep in suites.items()}
    R.print_report(results, log)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='store_true', help='только отчёт по сохранённым весам')
    ap.add_argument('--seeds', default=str(SEED), help='seed через запятую (первый - сохраняется)')
    ap.add_argument('--iters', type=int, default=None, help='итераций PPO')
    args = ap.parse_args()
    hw.configure_torch_threads()
    print(hw.describe())
    if hw.on_battery():
        print("Ноутбук работает от батареи - обучение будет заметно медленнее.")
    if args.eval:
        models = [load()]
    else:
        models = []
        with hw.keep_awake():
            for i, s in enumerate(int(x) for x in args.seeds.split(',')):
                ac = train_rl(seed=s, iters=args.iters, validate=True)
                if i == 0:
                    print(f"Сохранено: {save(ac)}")
                models.append((RL['FEAT'], ac))
    report(models)


if __name__ == '__main__':
    main()