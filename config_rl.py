"""
config_rl.py - конфигурация нового режима: обучение с подкреплением в цифровом двойнике с рандомизацией.
Прежний config.py не меняется; этот файл читают только rl_core.py и rl_trainer.py.
"""

import json
import os

import config as C
from config import A_CUR, C_RAW_FACTOR

_TUNED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuned')

RL = {
    'ALGO': os.environ.get('NIR_RL_ALGO') or 'ppo',
    'FEAT': os.environ.get('NIR_RL_FEAT') or 'u',
    'IMIT': {
        'ENABLED': (os.environ.get('NIR_RL_IMIT') or '1') == '1',
        'IN_PROXY': True,
        'EPISODES': 256,
        'EPOCHS': 15,
        'LR': 1e-3,
        'BATCH': 1024,
        'DR': True,
        'LOG_STD_AFTER': -1.6,
        'VF_WARMUP_ITERS': 3,
    },
    'REWARD': {
        'K_E': 2.0, 'K_ZONE': 2.0, 'K_OUT': 0.2, 'K_OUT_MAG': 0.5,
        'LAMBDA_U': 0.02, 'LAMBDA_DU': 0.05, 'BONUS': 0.02,
        'LAMBDA_REF': 0.5, 'LAMBDA_OS': 1.0,
    },
    'REF': {
        'MODE': os.environ.get('NIR_RL_REF') or 'tau',
        'TAU': (6.0, 30.0), 'OS': (0.0, 0.2),
        'PID_SCALE': (0.5, 1.5),
        'EVAL_TAU': (15.0, 0.05), 'EVAL_PID': (1.0, 1.0, 1.0),
    },
    'DR': {
        'A': (0.95, 0.995), 'C_FRAC': (0.05, 0.20),
        'DELAY': (1, 3), 'NOISE': (0.5, 5.0),
        'CROSS': 0.05, 'DEAD': (0.0, 0.05),
        'SAT_LO': (0.7, 1.0), 'SAT_HI': (0.7, 1.0),
    },
    'NOMINAL': {
        'A': A_CUR, 'C_FRAC': C_RAW_FACTOR, 'DELAY': 1, 'NOISE': 1.0, 'CROSS': 0.0,
        'DEAD': 0.0, 'SAT_LO': 1.0, 'SAT_HI': 1.0,
    },
    'DIST': {
        'P_KIND': {'none': 0.10, 'trend': 0.25, 'jump': 0.25, 'pulse': 0.10, 'sine': 0.10, 'multi': 0.20},
        'T0': (37, 90),
        'TREND': (0.005, 0.08), 'JUMP': (0.5, 3.0),
        'PULSE': (1.0, 4.0), 'PULSE_DUR': (3, 10),
        'SINE': (0.5, 1.5), 'SINE_PERIOD': (8, 24),
        'MULTI_N': (2, 4), 'MULTI_SCALE': 0.8,
    },
    'OOD': [
        {'name': 'a=0.93', 'A': 0.93},
        {'name': 'a=0.998', 'A': 0.998},
        {'name': 'c=0.03H', 'C_FRAC': 0.03},
        {'name': 'c=0.28H', 'C_FRAC': 0.28},
        {'name': 'delay=4', 'DELAY': 4},
        {'name': 'noise x8', 'NOISE': 8.0},
        {'name': 'cross 0.12', 'CROSS': 0.12},
        {'name': 'dead 0.1, sat 0.5', 'DEAD': 0.10, 'SAT_LO': 0.5},
    ],
    'PPO': {
        'N_ENVS': {'cuda': 1024, 'cpu': 64, 'xpu': 64}, # УВЕЛИЧЕНО ДЛЯ GPU
        'GAMMA': 0.98, 'LAMBDA': 0.95, 'CLIP': 0.2, 'EPOCHS': 4,
        'VF_COEF': 0.5, 'ENT_COEF': 0.0, 'MAX_GRAD': 0.5,
        'TARGET_KL': 0.03, 'LOG_STD_INIT': -0.7,
        'PRIV_CRITIC': True, 'CUDA_GRAPH': True, 'ITERS': 1000,
    },
    'HEAD': {'DU_MAX': 0.25, 'RES_MAX': 0.5},
    'METRIC': {'W_IAE': 1.0, 'W_TS': 3.0, 'W_OS': 5.0, 'W_EN': 0.5, 'W_FA': 2.0, 'W_ZE': 10.0, 'OS_FROM': 10},
    'SEARCH': {
        'PROXY_ITERS': 60, 'PROXY_SEEDS': 2, 'EVAL_EVERY': 5,
        'N_EVAL_DR': 48, 'N_OOD_EP': 4, 'N_QUIET': 4,
        'FULL_ITERS': 400, 'FINALISTS': 3, 'FINAL_SEEDS': (42, 43, 44),
        'MAX_PARAMS': 300_000, 'MAX_INFER_MS': 2.0, 'MAX_SIZE_MB': 1.5,
        'LIMIT_PENALTY': 1e3,
    },
    'SEEDS': {'EVAL': 160000, 'OOD': 170000, 'NOMINAL': 180000},
}

RL_ARCH = {
    'n_conv': 3, 'c1': 64, 'c2': 16, 'c3': 16, 'c4': 64,
    'k1': 7, 'k2': 5, 'k3': 3, 'k4': 3,
    'norm': 'none', 'temporal': 'avg', 'pool': 4, 'skip': 3, 'ctx_z': 0, 'ctx_type': 'conv',
    'fc': 128, 'head': 'direct', 'film': False,
}

# УВЕЛИЧЕН BATCH_SIZE ДЛЯ GPU
RL_TRAIN = {'LR': 3e-4, 'WEIGHT_DECAY': 1e-8, 'BATCH_SIZE': 8192}

RL['PID'] = {'mode': C.PID_INTEGRATOR, 'KP': C.KP, 'KI': C.KI, 'KD': C.KD,
             'WINDOW': C.PID_WINDOW, 'I_MAX': C.I_MAX, 'I_SEP': C.I_SEP}

_path = os.path.join(_TUNED_DIR, 'cnn_rl.json')
if os.path.exists(_path):
    with open(_path, encoding='utf-8') as _f:
        _tuned = json.load(_f)
    RL_ARCH.update(_tuned.get('model', {}))
    RL_TRAIN.update(_tuned.get('train', {}))

assert RL['ALGO'] in ('ppo',), "NIR_RL_ALGO: реализован только 'ppo'"
assert RL['FEAT'] in ('u', 'legacy'), "NIR_RL_FEAT должна быть 'u' или 'legacy'"
assert RL['REF']['MODE'] in ('tau', 'pid'), "NIR_RL_REF должна быть 'tau' или 'pid'"
