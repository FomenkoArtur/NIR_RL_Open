from dataclasses import dataclass

import numpy as np

from config import T_START
from metrics import compute_metrics, out_of_control_rate
from rl_control import run_scenario


@dataclass
class EpisodeResult:
    X: np.ndarray
    N: np.ndarray
    K: np.ndarray
    eps: np.ndarray
    u: np.ndarray
    ts: int
    iae: float
    idle_rate: float
    oc_rate: float


def simulate(scenario, seed, scale=1.0, model=None):
    episode = run_scenario(scenario, seed, scale, model)      # model - Controller (RL) или None
    ts, iae, idle_rate = compute_metrics(episode.eps, episode.u, T_START)
    oc_rate = out_of_control_rate(episode.N, T_START)
    return EpisodeResult(episode.X, episode.N, episode.K, episode.eps, episode.u,
                         ts, iae, idle_rate, oc_rate)