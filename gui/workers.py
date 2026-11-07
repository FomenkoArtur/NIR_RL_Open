import time

from PySide6.QtCore import QThread, Signal

import hw
from experiment import run_full_experiment
from rl_control import Controller
from rl_trainer import train_rl


class TrainWorker(QThread):
    iter_done = Signal(int, float, float, float)      # итерация, доход, KL, проверочная метрика J (nan - не считалась)
    message = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    SPEED_EVERY = 10          # как часто сообщать скорость обучения, итераций

    def __init__(self, iters, parent=None):
        super().__init__(parent)
        self._iters = int(iters)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            hw.configure_torch_threads()
            if hw.on_battery():
                self.message.emit("Ноутбук работает от батареи: процессор ограничивает мощность, "
                                  "обучение будет заметно медленнее. Подключите зарядку.")
            self.message.emit(f"Режим вычислений: {hw.describe()}")
            clock = {'t': time.perf_counter()}

            def on_iter(it, stats):
                val = stats.get('val_J')
                self.iter_done.emit(it, float(stats['ret']), float(stats['kl']), float('nan') if val is None else float(val))
                if it % self.SPEED_EVERY == 0:
                    now = time.perf_counter()
                    self.message.emit(f"Скорость: {(now - clock['t']) / self.SPEED_EVERY:.1f} с на итерацию")
                    clock['t'] = now

            # тот же train_rl, что и в main.py и rl_trainer.py: архитектура и LR/BATCH_SIZE берутся из config_rl
            with hw.keep_awake():
                ac = train_rl(iters=self._iters, log=self.message.emit, on_iter=on_iter,
                              should_stop=lambda: self._stop_requested, validate=True)

            if ac is None:
                self.failed.emit("Обучение прервано до завершения первой итерации.")
                return
            self.succeeded.emit(Controller(ac))
        except Exception as exc:
            self.failed.emit(str(exc))


class ExperimentWorker(QThread):
    succeeded = Signal(object, float)
    failed = Signal(str)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

    def run(self):
        try:
            hw.configure_torch_threads()
            frame, false_alarm = run_full_experiment(self._model)
            self.succeeded.emit(frame, float(false_alarm))
        except Exception as exc:
            self.failed.emit(str(exc))