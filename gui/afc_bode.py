"""
Вкладки АФХ (годограф Найквиста) и Боде (ЛАЧХ/ЛФЧХ) для замкнутого контура с RL (WP6).

Только numpy + PySide6 + matplotlib: импорт torch в gui/ запрещён, поэтому
частотная характеристика считается по номинальной дискретной модели объекта
из gui/nomenclature.py (x[k+1] = a*x[k] + b*u[k], инерция a = A_CUR). RL-регулятор
представлен эквивалентным постоянным коэффициентом Kc с единичной полосной
gain (точные веса нейросети без torch недоступны).

АФХ — годограф Найквиста замкнутого контура T(e^{jω}) = Kc·H / (1 + Kc·H).
Боде — ЛАЧХ 20·log10|T| и ЛФЧХ arg T против нормированной частоты ω ∈ [0, π].
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

import config
from plant import plant
from gui.nomenclature import items, nominal_params
from gui.plots import FigureWidget


class FrequencyResponseBase(FigureWidget):
    """Общая логика частотных характеристик: выбор номенклатуры + расчёт отклика."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo = QComboBox()
        for item in items():
            self.combo.addItem(item["name"], item["key"])
        self.combo.setCurrentIndex(0)
        self.combo.currentIndexChanged.connect(self.refresh)

        self.w = np.linspace(1e-4, np.pi, 1200)
        # self.refresh() убран из базового класса:
        # потомки создают оси (self.ax / self.ax_mag / self.ax_ph) уже после super().__init__()
        # и сами вызывают refresh() в конце своего конструктора.

    def control_row(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Номенклатура:"))
        row.addWidget(self.combo, 1)
        row.addStretch()
        return row

    def _responses(self, key):
        """Частотные характеристики для номенклатуры key."""
        params = nominal_params(key)
        k = float(params["k"])
        a = float(config.A_CUR)
        # Входной коэффициент каждого канала так, чтобы статический gain канала был k;
        # небольшая поправка на калиброванную ширину коридора S_i по каналам.
        b = k * (1.0 - a) * (plant.S / float(plant.S.mean()))
        z = np.exp(-1j * self.w)
        h = b[:, None] / (1.0 - a * z)[None, :]          # (8, W) открытый контур объекта

        # Эквивалентный коэффициент RL-регулятора: единичная полосная gain.
        kc = 1.0 / k if k > 0 else 1.0
        t = kc * h / (1.0 + kc * h)                      # замкнутый контур
        return self.w, h, t

    def refresh(self):
        raise NotImplementedError


class NyquistPlot(FrequencyResponseBase):
    """АФХ — годограф Найквиста замкнутого контура с RL-регулятором."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax = self.figure.subplots()
        self.ax.set_title(
            "Годограф Найквиста замкнутого контура T(e^{jω}) (контур с RL)",
            fontsize=14, pad=10,
        )
        self.ax.set_xlabel("Re T", fontsize=12)
        self.ax.set_ylabel("Im T", fontsize=12)
        self.refresh()

    def refresh(self):
        key = self.combo.currentData()
        w, h, t = self._responses(key)
        # Полный годограф: ω от 0 до 2π обходит единичный контур один раз
        # (вторая половина — комплексно-сопряжённая первая для вещественных коэффициентов).
        w_full = np.concatenate([self.w, 2.0 * np.pi - self.w[::-1]])
        t_full = np.concatenate([t, np.conj(t)[:, ::-1]], axis=1)

        self.ax.clear()
        for i in range(t_full.shape[0]):
            self.ax.plot(t_full[i].real, t_full[i].imag, color='tab:blue', alpha=0.25, lw=1)
        self.ax.plot(
            t_full[0].real, t_full[0].imag, color='tab:blue', lw=2.2,
            label=f"T = Kc·H / (1 + Kc·H), {self.combo.currentText()}",
        )
        self.ax.plot([-1.0], [0.0], 'rx', ms=10, mew=2, label="Критическая точка -1")
        self.ax.plot([1.0], [0.0], 'k.', ms=6, label="Точка 1 (ω = 0)")
        th = np.linspace(0, 2 * np.pi, 400)
        self.ax.plot(
            np.cos(th), np.sin(th), color='gray', ls=':', lw=1, alpha=0.6, label="Единичная окружность"
        )
        self.ax.axhline(0, color='k', lw=0.8, alpha=0.5)
        self.ax.axvline(0, color='k', lw=0.8, alpha=0.5)
        self.ax.grid(alpha=0.3)
        self.ax.tick_params(labelsize=11)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
        self.ax.text(
            0.98, 0.02,
            f"8 каналов (x1..x8), номенклатура: {self.combo.currentText()}",
            transform=self.ax.transAxes, fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', fc='white', alpha=0.7),
        )
        self.canvas.draw_idle()


class BodePlot(FrequencyResponseBase):
    """Боде — ЛАЧХ и ЛФЧХ замкнутого контура с RL-регулятором."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax_mag, self.ax_ph = self.figure.subplots(
            2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]}
        )
        self.ax_mag.set_title(
            "ЛАЧХ и ЛФЧХ замкнутого контура T(e^{jω}) (контур с RL)",
            fontsize=14, pad=10,
        )
        self.ax_mag.set_ylabel("|T|, дБ", fontsize=12)
        self.ax_ph.set_ylabel("∠T, град", fontsize=12)
        self.ax_ph.set_xlabel("Нормированная частота ω (рад/такт), 0 .. π", fontsize=12)
        self.refresh()

    def refresh(self):
        key = self.combo.currentData()
        w, h, t = self._responses(key)
        mag = 20.0 * np.log10(np.maximum(np.abs(t), 1e-12))
        ph = np.degrees(np.angle(t))

        self.ax_mag.clear()
        self.ax_ph.clear()
        for i in range(t.shape[0]):
            self.ax_mag.plot(w, mag[i], color='tab:blue', alpha=0.25, lw=1)
            self.ax_ph.plot(w, ph[i], color='tab:orange', alpha=0.25, lw=1)
        self.ax_mag.plot(
            w, mag[0], color='tab:blue', lw=2.2, label=f"|T|, {self.combo.currentText()}"
        )
        self.ax_ph.plot(w, ph[0], color='tab:orange', lw=2.2, label=f"∠T, {self.combo.currentText()}")
        self.ax_mag.axhline(0, color='k', lw=0.8, alpha=0.5)
        self.ax_ph.axhline(0, color='k', lw=0.8, alpha=0.5)
        for ax in (self.ax_mag, self.ax_ph):
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=11)
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        self.ax_ph.set_yticks([-180, -135, -90, -45, 0, 45, 90])
        self.canvas.draw_idle()