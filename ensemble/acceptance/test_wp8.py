import pytest

def test_reward_config():
    from config_rl import RL
    rw = RL['REWARD']
    assert rw.get('K_E') >= 5.0, "Штраф за ошибку K_E должен быть >= 5.0"
    assert 'LAMBDA_FA' in rw, "Нет штрафа за ложные срабатывания LAMBDA_FA"
    assert 'LAMBDA_STATIC' in rw, "Нет штрафа за статическую ошибку LAMBDA_STATIC"

def test_reward_logic():
    import inspect
    import rl_core
    src = inspect.getsource(rl_core.RLEnv._reward)
    assert 'FA_BAND' in src, "В _reward нет логики FA_BAND"
    assert 'LAMBDA_FA' in src, "В _reward нет логики LAMBDA_FA"
    assert 'LAMBDA_STATIC' in src, "В _reward нет логики LAMBDA_STATIC"
