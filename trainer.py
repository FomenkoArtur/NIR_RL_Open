"""
Функция потерь для предобучения RL-регулятора имитацией ПИД-эталона (её вызывает rl_core.PPO.pretrain_imitation).
Прежнее обучение CNN с учителем удалено; имя файла оставлено, чтобы rl_core.py не менять.
"""

import torch

from config import ACTIVE_W, FA_MARGIN, FA_W, QUIET_BAND


def weighted_mse(pred, target, active_w=ACTIVE_W):
    quiet = (target.abs() <= QUIET_BAND).float()
    err = (pred - target) ** 2 * (1.0 + active_w * (1.0 - quiet))
    false_alarm = (torch.relu(pred.abs() - FA_MARGIN) ** 2 * quiet).mean()
    return err.mean() + FA_W * false_alarm