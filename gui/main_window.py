from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

import numpy as np

import config
import hw
import mpc_control
import pid_control
import rl_control
import rl_trainer
from config import G, N_PARAMS
from config_rl import RL
from fastsim import episode_noise
from gui.afc_bode import BodePlot, NyquistPlot
from gui.labels import PARAMETERS, config_rows
from gui.nomenclature import items
from gui.plots import ChartPlot, SimulationPlot, TrainingPlot
from gui.simulation import EpisodeResult, simulate
from gui.workers import ExperimentWorker, TrainWorker
from metrics import compute_metrics, out_of_control_rate
from plant import plant
from scenarios import get_disturbance_profile


# === ОПЦИОНАЛЬНЫЙ МОДУЛЬ ИНДИКАТОРА СОСТОЯНИЯ ===
try:
    from gui.rpms_indicator import attach_rpms_indicator
except ImportError:
    attach_rpms_indicator = None


# =================================================



WEIGHTS_PATH = Path(rl_trainer.OUT_FILE)
SCENARIO_ORDER = ['S1', 'S2', 'S3', 'S4']
MODE_LABELS = {'rl': "С регулятором (RL)"}


class _TwinPlant:
    """Адаптер цифрового двойника (gui/plant.py) под интерфейс
    pid_control.PIDController / mpc_control.MPCController.

    Регуляторы действуют на нормализованные критерии N_i с уставкой G.
    Положительное u_i понижает x_i (формула evolve: -C_RAW * B * u), поэтому
    признак действия привода s_i = +1 и обратная связь отрицательная.
    """

    N_STATES = N_PARAMS

    def __init__(self):
        self.tau = np.ones(N_PARAMS)
        self.b = -np.eye(N_PARAMS)

    def setpoint(self, t):
        return np.full(N_PARAMS, float(G))

    def derivatives(self, x, u, t):
        return np.zeros(N_PARAMS)


def make_table(headers, rows=0):
    table = QTableWidget(rows, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def set_cell(table, row, column, text, align=Qt.AlignCenter):
    item = QTableWidgetItem(str(text))
    item.setTextAlignment(align | Qt.AlignVCenter)
    table.setItem(row, column, item)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система автоматического управления качеством продукции — апробация контура")
        # Увеличен размер окна для максимальной наглядности на презентации (Full HD)
        self.resize(1920, 1080)

        self.model = None
        self.results = {}
        self.scenario = None
        self.experiment_frame = None
        self.train_worker = None
        self.experiment_worker = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_frame)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_simulation_tab(), "Моделирование")
        self.tabs.addTab(self._build_chart_tab(), "Контрольные карты Шухарта")
        self.train_tab = self._build_training_tab()
        self.tabs.addTab(self.train_tab, "Обучение регулятора")
        self.tabs.addTab(self._build_experiment_tab(), "Таблица 1 (эксперимент)")
        self.tabs.addTab(self._build_parameters_tab(), "Параметры модели")

        # Частотные характеристики замкнутого контура с RL (вкладки АФХ и Боде).
        self.afc_plot = NyquistPlot()
        self.bode_plot = BodePlot()
        self.tabs.addTab(self._build_afc_tab(), "АФХ")
        self.tabs.addTab(self._build_bode_tab(), "Боде")

        # Адаптер цифрового двойника и эталоны ПИД/МПС для сравнительного моделирования.
        self._twin_plant = _TwinPlant()
        self.pid_controller = PIDController(
            self._twin_plant,
            kp=float(config.KP), ki=float(config.KI), kd=float(config.KD), u_max=1.0,
        )
        self.mpc_controller = MPCController(self._twin_plant, horizon=10)

        self.model_label = QLabel()
        self.statusBar().addPermanentWidget(self.model_label)
        self._set_model(None)
        self._try_load_default_weights()
        self._update_speed()

    def _build_simulation_tab(self):
        tab = QWidget()
        root = QHBoxLayout(tab)

        params_box = QGroupBox("Параметры прогона")
        form = QFormLayout(params_box)
        self.scenario_box = QComboBox()
        for key in SCENARIO_ORDER:
            self.scenario_box.addItem(f"{key} — {config.SCENARIOS[key]['description']}", key)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.5, 1.5)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setValue(1.0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(config.SEED)
        self.rl_check = QCheckBox(MODE_LABELS['rl'])
        self.rl_check.setChecked(True)
        self.animate_check = QCheckBox("Анимировать построение графика")
        self.animate_check.setChecked(True)
        self.run_button = QPushButton("Запустить моделирование")
        self.run_button.clicked.connect(self.run_simulation)

        self.nomenclature_box = QComboBox()
        for item in items():
            self.nomenclature_box.addItem(item["name"], item["key"])
        self.nomenclature_box.setCurrentIndex(0)

        form.addRow("Сценарий", self.scenario_box)
        form.addRow("Масштаб возмущения", self.scale_spin)
        form.addRow("Зерно ГСЧ (seed)", self.seed_spin)
        form.addRow("Номенклатура", self.nomenclature_box)
        form.addRow(self.rl_check)
        form.addRow(self.animate_check)
        form.addRow(self.run_button)

        playback_box = QGroupBox("Воспроизведение")
        playback = QVBoxLayout(playback_box)
        self.play_button = QPushButton("Воспроизвести")
        self.play_button.clicked.connect(self.toggle_playback)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, config.T_TOTAL)
        self.frame_slider.setValue(config.T_TOTAL)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 10)
        self.speed_slider.setValue(8)
        self.speed_slider.valueChanged.connect(self._update_speed)
        self.frame_label = QLabel(f"t = {config.T_TOTAL} / {config.T_TOTAL}")
        playback.addWidget(self.play_button)
        playback.addWidget(self.frame_label)
        playback.addWidget(self.frame_slider)
        playback.addWidget(QLabel("Скорость"))
        playback.addWidget(self.speed_slider)

        metrics_box = QGroupBox("Показатели качества")
        metrics_layout = QVBoxLayout(metrics_box)
        self.metrics_table = make_table(["Показатель", "Без рег.", "С RL"], 5)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setMaximumHeight(210)
        self.reduction_label = QLabel("Снижение IAE: —")
        metrics_layout.addWidget(self.metrics_table)
        metrics_layout.addWidget(self.reduction_label)

        self.save_plot_button = QPushButton("Сохранить график в PNG")
        self.save_plot_button.clicked.connect(self.save_simulation_plot)

        left = QVBoxLayout()
        left.addWidget(params_box)
        left.addWidget(playback_box)
        left.addWidget(metrics_box)

        # === ВСТРАИВАНИЕ ИНДИКАТОРА (если модуль включен в настройках) ===
        if attach_rpms_indicator is not None:
            self.rpms_indicator = attach_rpms_indicator(self, left)

        left.addWidget(self.save_plot_button)
        left.addStretch()
        # =================================================================

        left_widget = QWidget()
        left_widget.setLayout(left)
        # Ширина левой панели увеличена до комфортной (было 280, стало 360)
        left_widget.setFixedWidth(360)

        self.sim_plot = SimulationPlot()
        root.addWidget(left_widget)
        root.addWidget(self.sim_plot, 1)
        return tab

    def _build_chart_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.chart_param_box = QComboBox()
        for symbol, name, operation in PARAMETERS:
            self.chart_param_box.addItem(f"{symbol} — {name} ({operation})")
        self.chart_mode_box = QComboBox()
        for key, label in MODE_LABELS.items():
            self.chart_mode_box.addItem(label, key)
        self.chart_param_box.currentIndexChanged.connect(self._refresh_chart)
        self.chart_mode_box.currentIndexChanged.connect(self._refresh_chart)
        controls.addWidget(QLabel("Показатель:"))
        controls.addWidget(self.chart_param_box, 1)
        controls.addWidget(QLabel("Вариант контура:"))
        controls.addWidget(self.chart_mode_box)
        self.chart_plot = ChartPlot()
        layout.addLayout(controls)
        layout.addWidget(self.chart_plot, 1)
        return tab

    def _build_training_tab(self):
        tab = QWidget()
        root = QHBoxLayout(tab)
        left = QVBoxLayout()
        self.train_button = QPushButton("Обучить регулятор")
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Остановить обучение")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        self.load_button = QPushButton("Загрузить веса...")
        self.load_button.clicked.connect(self.load_weights_dialog)
        self.save_weights_button = QPushButton("Сохранить веса...")
        self.save_weights_button.clicked.connect(self.save_weights_dialog)
        self.iters_spin = QSpinBox()
        self.iters_spin.setRange(1, 100000)
        self.iters_spin.setValue(RL['PPO']['ITERS'])
        iters_row = QHBoxLayout()
        iters_row.addWidget(QLabel("Итераций PPO"))
        iters_row.addWidget(self.iters_spin, 1)
        self.hw_label = QLabel(self._hardware_text())
        self.hw_label.setWordWrap(True)
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 1)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        left.addLayout(iters_row)
        left.addWidget(self.hw_label)
        left.addWidget(self.train_button)
        left.addWidget(self.stop_button)
        left.addWidget(self.load_button)
        left.addWidget(self.save_weights_button)
        left.addWidget(self.train_progress)
        left.addWidget(self.train_log, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        # Ширина левой панели увеличена до комфортной (было 280, стало 360)
        left_widget.setFixedWidth(360)
        self.train_plot = TrainingPlot()
        root.addWidget(left_widget)
        root.addWidget(self.train_plot, 1)
        return tab

    def _build_experiment_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.experiment_button = QPushButton(
            f"Запустить эксперимент ({config.N_REAL} реализаций × 3 сценария)"
        )
        self.experiment_button.clicked.connect(self.start_experiment)
        self.export_button = QPushButton("Экспорт в CSV...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_experiment)
        self.experiment_progress = QProgressBar()
        self.experiment_progress.setRange(0, 1)
        controls.addWidget(self.experiment_button)
        controls.addWidget(self.export_button)
        controls.addWidget(self.experiment_progress, 1)
        self.experiment_table = make_table(
            ["Сценарий", "Особая причина изменчивости", "Ts, тактов (с регулятором)",
             "IAE без регулятора", "IAE с регулятором", "Снижение IAE, %",
             "Nᵢ вне [0,05; 0,95] без регулятора, %", "Nᵢ вне [0,05; 0,95] с регулятором, %"]
        )
        self.false_alarm_label = QLabel("S4: доля тактов с ложным срабатыванием регулятора: —")
        layout.addLayout(controls)
        layout.addWidget(self.experiment_table, 1)
        layout.addWidget(self.false_alarm_label)
        return tab

    def _build_parameters_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        charts_box = QGroupBox("Контрольные границы и веса частных критериев (калибровка цифрового двойника)")
        charts_layout = QVBoxLayout(charts_box)
        charts_table = make_table(["№", "Показатель", "Операция", "CL", "LCL", "UCL", "Вес w"], len(PARAMETERS))
        for i, (symbol, name, operation) in enumerate(PARAMETERS):
            set_cell(charts_table, i, 0, symbol)
            set_cell(charts_table, i, 1, name, Qt.AlignLeft)
            set_cell(charts_table, i, 2, operation, Qt.AlignLeft)
            set_cell(charts_table, i, 3, f"{plant.x_bar[i]:.4f}")
            set_cell(charts_table, i, 4, f"{plant.LCL[i]:.4f}")
            set_cell(charts_table, i, 5, f"{plant.UCL[i]:.4f}")
            set_cell(charts_table, i, 6, f"{plant.w[i]:.4f}")
        charts_layout.addWidget(charts_table)

        config_box = QGroupBox("Параметры эксперимента (config.py и config_rl.py, только чтение)")
        config_layout = QVBoxLayout(config_box)
        rows = config_rows()
        config_table = make_table(["Параметр", "Значение", "Описание"], len(rows))
        for i, (name, value, description) in enumerate(rows):
            set_cell(config_table, i, 0, name, Qt.AlignLeft)
            set_cell(config_table, i, 1, value)
            set_cell(config_table, i, 2, description, Qt.AlignLeft)
        config_layout.addWidget(config_table)

        layout.addWidget(charts_box, 1)
        layout.addWidget(config_box, 2)
        return tab

    def _build_afc_tab(self):
        """Вкладка «АФХ» — годограф Найквиста замкнутого контура с RL."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addLayout(self.afc_plot.control_row())
        layout.addWidget(self.afc_plot, 1)
        return tab

    def _build_bode_tab(self):
        """Вкладка «Боде» — ЛАЧХ и ЛФЧХ замкнутого контура с RL."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addLayout(self.bode_plot.control_row())
        layout.addWidget(self.bode_plot, 1)
        return tab

    def _set_model(self, model):
        self.model = model
        if model is None:
            self.model_label.setText("Регулятор: не обучен")
        else:
            self.model_label.setText("Регулятор: RL (PPO), готов")

    @staticmethod
    def _hardware_text():
        try:
            return f"Вычисления: {hw.describe()}"
        except Exception as exc:
            return f"Вычисления: устройство не определено ({exc})"

    def _try_load_default_weights(self):
        if WEIGHTS_PATH.exists() or Path(rl_trainer.COLAB_FILE).exists():
            self._load_weights(None, silent=True)

    def _load_weights(self, path, silent=False):
        try:
            model = rl_control.load_controller(path)
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Ошибка загрузки весов", str(exc))
            return
        self._set_model(model)
        self.train_log.appendPlainText(f"Загружены веса RL-регулятора: {path or WEIGHTS_PATH}")

    def _set_busy(self, busy, training=False):
        for widget in (self.run_button, self.train_button, self.load_button,
                       self.save_weights_button, self.experiment_button, self.iters_spin):
            widget.setEnabled(not busy)
        self.stop_button.setEnabled(busy and training)
        if busy:
            self.timer.stop()
            self.play_button.setText("Воспроизвести")

    def _require_model(self):
        if self.model is not None:
            return True
        QMessageBox.warning(
            self, "Регулятор не обучен",
            "Обучите регулятор или загрузите веса на вкладке «Обучение регулятора»."
        )
        self.tabs.setCurrentWidget(self.train_tab)
        return False

    def run_simulation(self):
        scenario = self.scenario_box.currentData()
        seed = self.seed_spin.value()
        scale = self.scale_spin.value()

        self.timer.stop()
        self.play_button.setText("Воспроизвести")
        results = {}
        # Базовый контур «без регулятора» считается всегда для сравнения в метриках.
        results['none'] = simulate(scenario, seed, scale)
        # RL первым: gui/rpms_indicator.py берёт results.get('cnn', первый элемент) - так индикатор следит за RL-регулятором
        if self.model is not None:
            results['rl'] = simulate(scenario, seed, scale, self.model)
        results['pid'] = self._run_twin(scenario, seed, scale, self.pid_controller)
        results['mpc'] = self._run_twin(scenario, seed, scale, self.mpc_controller)
        self.results = results
        self.scenario = scenario

        description = config.SCENARIOS[scenario]['description']
        self.sim_plot.show_results(results, scenario, description, scale, seed)
        self._fill_metrics()
        self._refresh_chart()

        if self.animate_check.isChecked():
            self.frame_slider.setValue(0)
            self.sim_plot.set_frame(0)
            self.timer.start()
            self.play_button.setText("Пауза")
        else:
            self.frame_slider.setValue(config.T_TOTAL)
            self.sim_plot.set_frame(config.T_TOTAL)

    def _run_twin(self, scenario, seed, scale, controller):
        """Эпизод цифрового двойника (gui/plant.py) под управлением controller
        (None — без управления) с тем же профилем возмущений и шумом, что у
        RL-симуляции. -> EpisodeResult."""
        rate, jump = get_disturbance_profile(scenario, config.T_START, scale)
        noise = episode_noise(seed)
        x = plant.x_bar.copy()
        if controller is not None:
            controller.reset()
        X = np.zeros((config.T_TOTAL, N_PARAMS))
        N = np.zeros((config.T_TOTAL, N_PARAMS))
        K = np.zeros(config.T_TOTAL)
        eps = np.zeros(config.T_TOTAL)
        U = np.zeros((config.T_TOTAL, N_PARAMS))
        normalizer = plant.make_normalizer()
        for t in range(config.T_TOTAL):
            if t == config.T_START:
                x = x + jump
            n = normalizer.update(x)
            N[t] = n
            eps[t] = G - float(n.mean())
            K[t] = float(plant.compute_super_criterion(n))
            u = np.zeros(N_PARAMS) if controller is None else np.asarray(controller.act(n, t), dtype=float)
            U[t] = u
            X[t] = x
            if t < config.T_TOTAL - 1:
                x = plant.evolve(x, rate[t], u, noise[t])
        ts, iae, idle_rate = compute_metrics(eps, U, config.T_START)
        oc_rate = out_of_control_rate(N, config.T_START)
        return EpisodeResult(X, N, K, eps, U, ts, iae, idle_rate, oc_rate)

    def _fill_metrics(self):
        rows = ["Ts, тактов", "IAE", "Доля тактов |u| > порога, %", "Макс. |ΔK| после t0",
                "Доля тактов с Nᵢ вне [0,05; 0,95], %"]
        for row, name in enumerate(rows):
            set_cell(self.metrics_table, row, 0, name, Qt.AlignLeft)
            for column, mode in ((1, 'none'), (2, 'rl')):
                result = self.results.get(mode)
                set_cell(self.metrics_table, row, column, self._metric_text(result, row))
        none, rl = self.results.get('none'), self.results.get('rl')
        if none is not None and rl is not None and none.iae > 0:
            reduction = 100 * (1 - rl.iae / none.iae)
            self.reduction_label.setText(f"Снижение IAE: {reduction:.1f} %")
        else:
            self.reduction_label.setText("Снижение IAE: —")

    @staticmethod
    def _metric_text(result, row):
        if result is None:
            return "—"
        if row == 0:
            return f"> {config.T_OBS}" if result.ts >= config.T_OBS else str(result.ts)
        if row == 1:
            return f"{result.iae:.2f}"
        if row == 2:
            return f"{100 * result.idle_rate:.1f}"
        if row == 3:
            window = abs(result.K[config.T_START:config.T_START + config.T_OBS] - config.G)
            return f"{window.max():.3f}"
        return f"{100 * result.oc_rate:.0f}"

    def _refresh_chart(self):
        if not self.results:
            return
        mode = self.chart_mode_box.currentData()
        result = self.results.get(mode)
        if result is None:
            mode, result = next(iter(self.results.items()))
        self.chart_plot.show_chart(result, self.chart_param_box.currentIndex(), MODE_LABELS[mode])

    def toggle_playback(self):
        if not self.results:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Воспроизвести")
            return
        if self.frame_slider.value() >= config.T_TOTAL:
            self.frame_slider.setValue(0)
        self.timer.start()
        self.play_button.setText("Пауза")

    def _advance_frame(self):
        value = self.frame_slider.value() + 1
        if value >= config.T_TOTAL:
            self.timer.stop()
            self.play_button.setText("Воспроизвести")
        self.frame_slider.setValue(min(value, config.T_TOTAL))

    def _on_frame_changed(self, value):
        self.frame_label.setText(f"t = {value} / {config.T_TOTAL}")
        if self.results:
            self.sim_plot.set_frame(value)

    def _update_speed(self):
        self.timer.setInterval(210 - 20 * self.speed_slider.value())

    def save_simulation_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "переходный_процесс.png", "PNG (*.png)"
        )
        if path:
            self.sim_plot.save(path)
            self.statusBar().showMessage(f"График сохранён: {path}", 5000)

    def start_training(self):
        self.train_log.clear()
        self.train_plot.reset()
        self.train_progress.setRange(0, self.iters_spin.value())
        self.train_progress.setValue(0)
        self._set_busy(True, training=True)
        worker = TrainWorker(self.iters_spin.value(), self)
        worker.iter_done.connect(self._on_iter_done)
        worker.message.connect(self.train_log.appendPlainText)
        worker.succeeded.connect(self._on_training_succeeded)
        worker.failed.connect(self._on_training_failed)
        self.train_worker = worker
        worker.start()

    def stop_training(self):
        if self.train_worker is not None:
            self.train_worker.request_stop()
            self.stop_button.setEnabled(False)

    def _on_iter_done(self, iteration, ret, kl, val_j):
        checked = val_j == val_j                      # nan - проверка в эту итерацию не считалась
        self.train_plot.add_point(iteration, ret, val_j if checked else None)
        self.train_progress.setValue(iteration)
        self.train_log.appendPlainText(
            f"Итерация {iteration}: доход = {ret:.3f}, KL = {kl:.4f}" + (f", J (проверка) = {val_j:.3f}" if checked else "")
        )

    def _on_training_succeeded(self, model):
        rl_trainer.save(model.ac, WEIGHTS_PATH)
        self._set_model(model)
        self.train_log.appendPlainText(f"Обучение завершено. Веса сохранены: {WEIGHTS_PATH}")
        self._finish_training()

    def _on_training_failed(self, message):
        self.train_log.appendPlainText(f"Ошибка: {message}")
        self._finish_training()

    def _finish_training(self):
        self.train_progress.setRange(0, 1)
        self._set_busy(False)
        self.train_worker = None

    def load_weights_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить веса RL-регулятора", str(WEIGHTS_PATH.parent), "PyTorch (*.pt *.pth)")
        if path:
            self._load_weights(path)

    def save_weights_dialog(self):
        if self.model is None:
            QMessageBox.information(self, "Нет весов", "Регулятор ещё не обучен.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить веса", str(WEIGHTS_PATH), "PyTorch (*.pt)")
        if path:
            rl_trainer.save(self.model.ac, path)
            self.statusBar().showMessage(f"Веса сохранены: {path}", 5000)

    def start_experiment(self):
        if not self._require_model():
            return
        self.experiment_progress.setRange(0, 0)
        self._set_busy(True)
        worker = ExperimentWorker(self.model, self)
        worker.succeeded.connect(self._on_experiment_succeeded)
        worker.failed.connect(self._on_experiment_failed)
        self.experiment_worker = worker
        worker.start()

    def _on_experiment_succeeded(self, frame, false_alarm):
        self.experiment_frame = frame
        self.experiment_table.setRowCount(len(frame))
        for row in range(len(frame)):
            for column in range(frame.shape[1]):
                align = Qt.AlignLeft if column == 1 else Qt.AlignCenter
                set_cell(self.experiment_table, row, column, frame.iat[row, column], align)
        self.false_alarm_label.setText(
            f"S4: доля тактов с ложным срабатыванием регулятора: {false_alarm:.2f} %"
        )
        self.export_button.setEnabled(True)
        self._finish_experiment()

    def _on_experiment_failed(self, message):
        QMessageBox.critical(self, "Ошибка эксперимента", message)
        self._finish_experiment()

    def _finish_experiment(self):
        self.experiment_progress.setRange(0, 1)
        self._set_busy(False)
        self.experiment_worker = None

    def export_experiment(self):
        if self.experiment_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт таблицы", "Таблица_1.csv", "CSV (*.csv)")
        if path:
            self.experiment_frame.to_csv(path, index=False, sep=';', encoding='utf-8-sig')
            self.statusBar().showMessage(f"Таблица сохранена: {path}", 5000)

    def closeEvent(self, event):
        if self.train_worker is not None:
            self.train_worker.request_stop()
            self.train_worker.wait()
        if self.experiment_worker is not None:
            self.experiment_worker.wait()
        super().closeEvent(event)