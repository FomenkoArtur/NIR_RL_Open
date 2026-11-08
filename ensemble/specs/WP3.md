\# WP3: CNN 8→8 без суперкритерия, удаление TuneCNN



\## Назначение

Переделать архитектуру CNN-регулятора: вход — 8 нормализованных показателей,

выход — 8 управляющих воздействий. Суперкритерий K(t) и веса w\_i исключены

из обучения. TuneCNN удалён.



\## Ссылки на работу

2.2.2, 3.1.2.



\## Файлы

\- `config.py` — `N\_FEATURES = 8` (было 17).

\- `rl\_core.py` — `RLNet`, `ActorCritic`, `RLEnv.window()`.

\- `trainer.py` — функция потерь без суперкритерия.

\- `tune\_cnn.py` — удалить `TuneCNN` и `suggest\_rl` (поиск архитектуры не нужен).

\- `episode.py` — `CNNPolicy` (если используется), вход 8 каналов.



\## Требования

1\. `config.N\_FEATURES == 8`. `config.N\_PARAMS == 8`.

2\. `ActorCritic(RL\_ARCH).actor.fc2.out\_features == 8`.

3\. `RLEnv.window(feat)` возвращает тензор `(B, 8, 32)`.

4\. В `rl\_core.py`, `rl\_trainer.py`, `rl\_control.py`, `episode.py`, `trainer.py`:

&#x20;  - нет `supercrit`, `superkrit`, `supercriterion`, `ahp\_w`, `K\_total`;

&#x20;  - нет `w32` в `RLEnv.window` (убрать признак w\_i\*N\_i).

5\. `tune\_cnn.py`:

&#x20;  - нет класса `TuneCNN`;

&#x20;  - нет функции `suggest\_rl`.

6\. Residual connection — по желанию (не проверяется).



\## Запреты

\- НЕ менять публичный API `run\_policy`, `RLPolicy`, `PPO`, `make\_suites`, `ActorCritic`.

\- НЕ запускать обучение внутри тестов — только структура.

\- НЕ менять `plant.py`, `plant\_equations.py`, `scenarios.py`.



\## Критерии приёмки

См. `ensemble/acceptance/test\_wp3.py` (только структура, быстро).

