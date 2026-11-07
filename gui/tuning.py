"""
Вкладки «Подбор архитектуры» (tune_cnn.py) и «Подбор ПИД» (tune_pid.py): запуск, остановка, живой график, лог.
Подбор идёт в отдельном процессе; GUI читает его вывод, а для архитектуры - ещё и базу Optuna (tuned/cnn_rl_study.db).
Анимация спокойная: новые точки плавно проявляются, линия лучшего значения скользит к новому уровню, последняя точка мягко пульсирует.
"""

import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import psutil
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

import hw_tune
from gui.plots import BlitManager, FigureWidget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE, ORANGE, GREEN, GRAY = 'tab:blue', 'tab:orange', 'tab:green', '0.6'


def ease(x):
    """Плавное замедление к концу: 0 -> 1."""
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


class Glide:
    """Показанные значения плавно догоняют целевые; новые элементы стартуют с уровня предыдущего."""

    def __init__(self):
        self.shown = np.zeros(0)
        self.target = np.zeros(0)

    def set_target(self, values):
        values = np.asarray(values, dtype=float)
        shown = values.copy()
        n = min(len(self.shown), len(values))
        shown[:n] = self.shown[:n]
        if 0 < n < len(values):
            shown[n:] = shown[n - 1]
        self.shown, self.target = shown, values

    def step(self, k=0.16):
        if len(self.target):
            self.shown += (self.target - self.shown) * k
        return self.shown


class ProcessRunner(QThread):
    """Запускает скрипт подбора, отдаёт его вывод построчно. Остановка: сначала мягко (файл), потом дерево процессов."""
    line = Signal(str)
    finished_with = Signal(int)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self._args = args
        self._proc = None

    def run(self):
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        flags = 0x08000000 if sys.platform == 'win32' else 0          # без чёрного окна консоли
        try:
            self._proc = subprocess.Popen([sys.executable, '-u'] + self._args, cwd=ROOT, env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                          encoding='utf-8', errors='replace', creationflags=flags)
            for text in self._proc.stdout:
                self.line.emit(text.rstrip())
            self.finished_with.emit(self._proc.wait())
        except Exception as exc:
            self.line.emit(f"Не удалось запустить: {exc}")
            self.finished_with.emit(-1)

    def request_stop(self):
        os.makedirs(os.path.dirname(hw_tune.STOP_FILE), exist_ok=True)
        open(hw_tune.STOP_FILE, 'w').close()                           # скрипты проверяют файл между шагами
        QTimer.singleShot(30000, self.kill)                            # не успели за 30 с - убираем процессы

    def kill(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.Error:
            pass


# ------------------------------------------------ графики ------------------------------------------------
class TrialsPlot(FigureWidget):
    """Сверху - стоимость J каждого испытания и лучшее значение; снизу - промежуточные значения Hyperband свежих испытаний."""

    RECENT = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blit = BlitManager(self.canvas)
        self.ax_c, self.ax_i = self.figure.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]})
        self.reset()

    def reset(self):
        for ax in (self.ax_c, self.ax_i):
            ax.clear()
            ax.grid(alpha=0.3, which='both')
            ax.tick_params(labelsize=12)
        self.ax_c.set_title("Подбор архитектуры RL-регулятора (Optuna: TPE + Hyperband)", fontsize=16)
        self.ax_c.set_ylabel("Метрика J (меньше - лучше)", fontsize=14)
        self.ax_c.set_xlabel("Номер испытания", fontsize=14)
        self.ax_c.set_yscale('log')
        self.ax_i.set_ylabel("J по ходу обучения", fontsize=14)
        self.ax_i.set_xlabel("Итерация PPO", fontsize=14)
        self.ax_i.set_yscale('log')
        self._births, self._glide = {}, Glide()
        self._done_n, self._done_y, self._cut_n, self._cut_y, self._best_x = [], [], [], [], []
        self._lines = []                                               # (линия, номер испытания)
        self.dots = self.ax_c.scatter([], [], s=[], color=BLUE, alpha=0.85, label='Завершённые')
        self.cut = self.ax_c.scatter([], [], s=[], marker='x', color=GRAY, label='Отсечённые')
        self.best_line, = self.ax_c.plot([], [], color=ORANGE, lw=2.6, drawstyle='steps-post', label='Лучшее значение')
        self.halo, = self.ax_c.plot([], [], 'o', mfc='none', mec=GREEN, mew=2, ms=14)
        self.ax_c.legend(loc='upper right', fontsize=12, framealpha=0.9)
        self._refresh_artists()
        self.canvas.draw()

    def _refresh_artists(self):
        arts = [self.dots, self.cut, self.best_line, self.halo] + [ln for ln, _ in self._lines]
        for art in arts:
            art.set_animated(True)
        self._blit.artists = arts

    def update_data(self, rows):
        """rows: [{'n', 'state', 'value', 'inter': [(шаг, значение)]}] из базы Optuna."""
        now = time.time()
        done = sorted((r for r in rows if r['state'] == 'COMPLETE' and r['value'] is not None), key=lambda r: r['n'])
        cut = [r for r in rows if r['state'] == 'PRUNED' and r['inter']]
        for r in done + cut:
            self._births.setdefault(r['n'], now)
        self._done_n, self._done_y = [r['n'] for r in done], [r['value'] for r in done]
        self._cut_n, self._cut_y = [r['n'] for r in cut], [r['inter'][-1][1] for r in cut]
        self._best_x = self._done_n
        self._glide.set_target(np.minimum.accumulate(self._done_y) if self._done_y else [])

        for ln, _ in self._lines:
            ln.remove()
        self._lines = []
        best_n = done[int(np.argmin(self._done_y))]['n'] if done else None
        shown = [r for r in rows if r['inter']][-self.RECENT:]
        for r in shown:
            steps, vals = zip(*r['inter'])
            is_best = r['n'] == best_n
            ln, = self.ax_i.plot(steps, vals, color=ORANGE if is_best else BLUE, lw=2.4 if is_best else 1.4,
                                 alpha=0.0, zorder=3 if is_best else 2)
            self._lines.append((ln, r['n']))
            self._births.setdefault(('i', r['n']), now)
        for ax in (self.ax_c, self.ax_i):
            ax.relim()
            ax.autoscale_view()
        self._refresh_artists()
        self.canvas.draw()

    def tick(self):
        now = time.time()
        grow = lambda n: 46 * ease((now - self._births.get(n, 0)) / 0.9)
        if self._done_n:
            self.dots.set_offsets(np.c_[self._done_n, self._done_y])
            self.dots.set_sizes([grow(n) for n in self._done_n])
            self.best_line.set_data(self._best_x, self._glide.step())
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * now / 2.6)      # медленное «дыхание» последней точки
            self.halo.set_data([self._done_n[-1]], [self._done_y[-1]])
            self.halo.set_markersize(13 + 4 * pulse)
            self.halo.set_markeredgecolor((0.17, 0.63, 0.17, 0.25 + 0.55 * pulse))
        if self._cut_n:
            self.cut.set_offsets(np.c_[self._cut_n, self._cut_y])
            self.cut.set_sizes([grow(n) * 0.8 for n in self._cut_n])
        for ln, n in self._lines:
            ln.set_alpha((0.95 if ln.get_color() == ORANGE else 0.45) * ease((now - self._births.get(('i', n), 0)) / 1.2))
        self._blit.update()


class PidPlot(FigureWidget):
    """Сверху - сходимость CMA-ES (лучший, среднее, полоса лучший-худший); снизу - коэффициенты: ручные и найденные."""

    NAMES = ['Kp', 'Ki', 'Kd', 'LI (окно)']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blit = BlitManager(self.canvas)
        self.ax_g, self.ax_p = self.figure.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]})
        self.reset()

    def reset(self):
        for ax in (self.ax_g, self.ax_p):
            ax.clear()
            ax.grid(alpha=0.3, which='both')
            ax.tick_params(labelsize=12)
        self.ax_g.set_title("Подбор ПИД-эталона (CMA-ES по цифровому двойнику)", fontsize=16)
        self.ax_g.set_ylabel("Функция качества (меньше - лучше)", fontsize=14)
        self.ax_g.set_xlabel("Поколение", fontsize=14)
        self.ax_g.set_yscale('log')
        self.ax_p.set_xscale('log')
        self.ax_p.set_xlim(1e-3, 64)
        self.ax_p.set_ylim(-0.7, 3.7)
        self.ax_p.set_yticks(range(4))
        self.ax_p.set_yticklabels(self.NAMES)
        self.ax_p.invert_yaxis()
        self.ax_p.set_xlabel("Значение коэффициента (логарифмическая шкала)", fontsize=14)
        self._gens = []
        self._best, self._mean, self._base = Glide(), Glide(), None
        self._band = None
        self._manual = self._found = None
        self._pshown = {'manual': np.full(4, 1e-3), 'found': np.full(4, 1e-3)}
        self.best_line, = self.ax_g.plot([], [], color=ORANGE, lw=2.6, marker='o', ms=5, label='Лучший в поколении')
        self.mean_line, = self.ax_g.plot([], [], color=BLUE, lw=1.8, alpha=0.8, label='Среднее по популяции')
        self.base_line = self.ax_g.axhline(np.nan, color=GRAY, ls='--', lw=1.8, label='Ручные значения')
        self.halo, = self.ax_g.plot([], [], 'o', mfc='none', mec=GREEN, mew=2, ms=14)
        self.ax_g.legend(loc='upper right', fontsize=12, framealpha=0.9)
        self.lolli = {}
        for key, color, dy in (('manual', GRAY, -0.17), ('found', BLUE, 0.17)):
            lines = [self.ax_p.plot([], [], color=color, lw=3, solid_capstyle='round')[0] for _ in range(4)]
            dots = self.ax_p.plot([], [], 'o', color=color, ms=9, label='Ручные' if key == 'manual' else 'Найденные')[0]
            self.lolli[key] = (lines, dots, dy)
        self.ax_p.legend(loc='lower right', fontsize=12, framealpha=0.9)
        self._refresh_artists()
        self.canvas.draw()

    def _refresh_artists(self):
        arts = [self.best_line, self.mean_line, self.halo, self.base_line]
        if self._band is not None:
            arts.append(self._band)
        for lines, dots, _ in self.lolli.values():
            arts += lines + [dots]
        for art in arts:
            art.set_animated(True)
        self._blit.artists = arts

    def set_manual(self, params, cost):
        self._manual, self._base = np.array(params, float), cost
        self.base_line.set_ydata([cost, cost])
        self._rescale()

    def add_generation(self, gen, best, mean, worst):
        self._gens.append((gen, best, mean, worst))
        g = np.array(self._gens)
        self._best.set_target(g[:, 1])
        self._mean.set_target(g[:, 2])
        if self._band is not None:
            self._band.remove()
        self._band = self.ax_g.fill_between(g[:, 0], g[:, 1], g[:, 3], color=BLUE, alpha=0.12, linewidth=0)
        self._rescale()

    def set_found(self, params):
        self._found = np.array(params, float)

    def _rescale(self):
        self.ax_g.relim()
        self.ax_g.autoscale_view()
        self._refresh_artists()
        self.canvas.draw()

    def tick(self):
        now = time.time()
        if self._gens:
            gens = [g[0] for g in self._gens]
            self.best_line.set_data(gens, self._best.step())
            self.mean_line.set_data(gens, self._mean.step())
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * now / 2.6)
            self.halo.set_data([gens[-1]], [self._best.shown[-1]])
            self.halo.set_markersize(13 + 4 * pulse)
            self.halo.set_markeredgecolor((0.17, 0.63, 0.17, 0.25 + 0.55 * pulse))
        for key, values in (('manual', self._manual), ('found', self._found)):
            lines, dots, dy = self.lolli[key]
            if values is None:
                continue
            logs = np.log10(self._pshown[key])                          # скольжение в логарифмической шкале
            logs += (np.log10(values) - logs) * 0.12
            self._pshown[key] = 10 ** logs
            for i, line in enumerate(lines):
                line.set_data([1e-3, self._pshown[key][i]], [i + dy, i + dy])
            dots.set_data(self._pshown[key], [i + dy for i in range(4)])
        self._blit.update()


# ------------------------------------------------ вкладки ------------------------------------------------
class _TuneTab(QWidget):
    """Общий каркас: слева параметры, кнопки, прогресс и лог; справа график."""

    def __init__(self, plot, title, summary, parent=None):
        super().__init__(parent)
        self.plot = plot
        self.runner = None
        self.start_button = QPushButton("Запустить подбор")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.summary = QLabel(summary)
        self.summary.setWordWrap(True)
        self.params_form = QVBoxLayout()
        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)
        box_layout.addLayout(self.params_form)
        left = QVBoxLayout()
        left.addWidget(box)
        left.addWidget(self.summary)
        left.addWidget(self.start_button)
        left.addWidget(self.stop_button)
        left.addWidget(self.progress)
        left.addWidget(self.log, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(360)
        root = QHBoxLayout(self)
        root.addWidget(left_widget)
        root.addWidget(plot, 1)
        self.anim = QTimer(self)                                       # ~25 кадров/с, только пока идёт подбор или анимация
        self.anim.setInterval(40)
        self.anim.timeout.connect(self.plot.tick)

    def add_spin(self, label, low, high, value, special=None):
        row = QHBoxLayout()
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        if special:
            spin.setSpecialValueText(special)
        row.addWidget(QLabel(label))
        row.addWidget(spin, 1)
        self.params_form.addLayout(row)
        return spin

    def _set_running(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def launch(self, args):
        self.log.clear()
        hw_tune.clear_stop()
        self.runner = ProcessRunner(args, self)
        self.runner.line.connect(self.on_line)
        self.runner.finished_with.connect(self.on_done)
        self._set_running(True)
        self.anim.start()
        self.runner.start()

    def stop(self):
        if self.runner is not None:
            self.stop_button.setEnabled(False)
            self.log.appendPlainText("Останавливаю: дожидаюсь конца текущего шага...")
            self.runner.request_stop()

    def on_done(self, code):
        self.log.appendPlainText("Готово." if code == 0 else f"Процесс завершился с кодом {code}.")
        self._set_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if code == 0 else 0)
        QTimer.singleShot(4000, self.anim.stop)                        # дать анимации спокойно закончиться
        self.runner = None

    def shutdown(self):
        if self.runner is not None:
            self.runner.request_stop()
            self.runner.kill()
            self.runner.wait(3000)

    def start(self):
        raise NotImplementedError

    def on_line(self, text):
        self.log.appendPlainText(text)


class TuneRlTab(_TuneTab):
    def __init__(self, parent=None):
        super().__init__(TrialsPlot(), "Параметры подбора", TuneRlTab._hardware_text(), parent)
        self.trials_spin = self.add_spin("Испытаний", 1, 100000, 60)
        self.workers_spin = self.add_spin("Воркеров", 0, 64, 0, special="Авто")
        self.poll = QTimer(self)
        self.poll.setInterval(2000)
        self.poll.timeout.connect(self.read_study)
        self._base, self._total = 0, 0

    @staticmethod
    def _hardware_text():
        try:
            s = hw_tune.settings()
            return (f"Устройства: {', '.join(s['devices'])}; воркеров {s['workers']} "
                    f"({s['workers_per_device']} на устройство), потоков на воркер {s['threads']} [{s['source']}]. "
                    f"Оптимум для этого компьютера - python hw_tune.py.")
        except Exception as exc:
            return f"Устройства не определены: {exc}"

    @staticmethod
    def _read_rows():
        """Испытания из базы Optuna (пусто, пока подбор не создал базу)."""
        try:
            import optuna
            import tune_cnn
            optuna.logging.set_verbosity(optuna.logging.ERROR)
            study = optuna.load_study(study_name=tune_cnn.STUDY_NAME, storage=tune_cnn.STORAGE_URL)
            return [{'n': t.number, 'state': t.state.name, 'value': t.value,
                     'inter': sorted(t.intermediate_values.items())} for t in study.get_trials(deepcopy=False)]
        except Exception:
            return []

    def start(self):
        rows = self._read_rows()
        self._base = sum(1 for r in rows if r['state'] in ('COMPLETE', 'PRUNED', 'FAIL'))
        self._total = self.trials_spin.value()
        self.plot.reset()
        self.progress.setRange(0, self._total)
        self.progress.setValue(0)
        args = ['tune_cnn.py', str(self._total)] + ([str(self.workers_spin.value())] if self.workers_spin.value() else [])
        self.launch(args)
        self.poll.start()
        self.read_study()

    def read_study(self):
        rows = self._read_rows()
        finished = sum(1 for r in rows if r['state'] in ('COMPLETE', 'PRUNED', 'FAIL')) - self._base
        self.progress.setValue(max(0, min(finished, self._total)))
        self.plot.update_data(rows)

    def on_done(self, code):
        self.poll.stop()
        self.read_study()
        super().on_done(code)
        if code == 0:
            self.log.appendPlainText("Архитектура сохранена в tuned/cnn_rl.json, веса - tuned/controller_rl.pt. "
                                     "Новая архитектура подхватится при следующем запуске программы.")


class TunePidTab(_TuneTab):
    RE_MANUAL = re.compile(r'Ручные значения \(([^)]*)\): cost = ([\d.eE+-]+)')
    RE_GEN = re.compile(r'поколение (\d+): лучший cost = ([\d.eE+-]+) \(среднее ([\d.eE+-]+), худший ([\d.eE+-]+)\)')
    RE_FOUND = re.compile(r'Найдено: Kp=([\d.eE+-]+) Ki=([\d.eE+-]+) Kd=([\d.eE+-]+) LI=(\d+)')

    def __init__(self, parent=None):
        super().__init__(PidPlot(), "Параметры подбора",
                         f"Расчёт на процессоре (numpy). Процессов по умолчанию: {hw_tune.pid_procs()}; "
                         f"оптимум для этого компьютера - python hw_tune.py.", parent)
        self.procs_spin = self.add_spin("Процессов", 0, 128, 0, special="Авто")
        self._max_gen = 60

    def start(self):
        self.plot.reset()
        self.progress.setRange(0, self._max_gen)
        self.progress.setValue(0)
        self.launch(['tune_pid.py'] + ([str(self.procs_spin.value())] if self.procs_spin.value() else []))

    def on_line(self, text):
        super().on_line(text)
        if m := self.RE_MANUAL.search(text):
            self.plot.set_manual([float(x) for x in m.group(1).split(',')], float(m.group(2)))
        elif m := self.RE_GEN.search(text):
            gen = int(m.group(1))
            self.plot.add_generation(gen, float(m.group(2)), float(m.group(3)), float(m.group(4)))
            self.progress.setValue(min(gen, self._max_gen))
        elif m := self.RE_FOUND.search(text):
            self.plot.set_found([float(m.group(i)) for i in range(1, 5)])


def add_tuning_tabs(window):
    """Добавляет две вкладки в главное окно (вызывается из run_gui.py после создания MainWindow)."""
    tabs = [TuneRlTab(), TunePidTab()]
    window.tabs.addTab(tabs[0], "Подбор архитектуры")
    window.tabs.addTab(tabs[1], "Подбор ПИД")
    window.tuning_tabs = tabs
    QApplication.instance().aboutToQuit.connect(lambda: [t.shutdown() for t in tabs])