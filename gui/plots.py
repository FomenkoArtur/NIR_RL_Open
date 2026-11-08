import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from config import A2, D4, DELTA, G, IDLE_THRESHOLD, N_HIGH, N_LOW, T_START, T_TOTAL
from gui.labels import PARAMETERS, zone_labels
from plant import plant

MODE_STYLE = {
    'rl': ('RL (Свёрточная сеть)', 'tab:blue', 3.8),
    'pid': ('PID-регулятор', 'tab:green', 2.8),
    'mpc': ('MPC (Предсказание)', 'tab:orange', 2.8),
}


def fit_ylim(ax, values, include=(), pad_lo=0.05, pad_hi=0.05):
    values = np.asarray(values, dtype=float)
    lo = min([values.min(), *include])
    hi = max([values.max(), *include])
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - pad_lo * span, hi + pad_hi * span)


class FigureWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(16, 10), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def save(self, path):
        self.figure.savefig(path, dpi=300, bbox_inches='tight')


class BlitManager:
    def __init__(self, canvas):
        self.canvas = canvas
        self.artists = []
        self._background = None
        canvas.mpl_connect('draw_event', self._on_draw)

    def _on_draw(self, event):
        self._background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_artists()

    def _draw_artists(self):
        for artist in self.artists:
            self.canvas.figure.draw_artist(artist)

    def update(self):
        if self._background is None:
            self.canvas.draw()
            return
        self.canvas.restore_region(self._background)
        self._draw_artists()
        self.canvas.blit(self.canvas.figure.bbox)


class SimulationPlot(FigureWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._blit = BlitManager(self.canvas)
        self.ax_k, self.ax_u, self.ax_n = self.figure.subplots(
            3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2.5, 2.5]}
        )
        self._series = []
        self._decorate_empty()

    def _decorate_empty(self):
        self.ax_k.set_title("Запустите моделирование")
        self.canvas.draw_idle()

    def show_results(self, results, scenario, description, scale, seed):
        for ax in (self.ax_k, self.ax_u, self.ax_n):
            ax.clear()
        self._series = []
        t = np.arange(T_TOTAL)

        self.ax_k.axhline(G, color='k', alpha=0.4, label=f"Программное задание g = {G}")
        self.ax_k.axhspan(G - DELTA, G + DELTA, color='green', alpha=0.12, label=f"Допустимая полоса ±{DELTA}")
        if scenario != 'S4':
            for ax in (self.ax_k, self.ax_u, self.ax_n):
                ax.axvline(T_START, color='gray', ls=':')

        k_values = [G - DELTA, G + DELTA]

        # Отрисовка K(t) для всех трех регуляторов
        for mode in ('rl', 'pid', 'mpc'):
            if mode not in results or mode not in MODE_STYLE:
                continue
            result = results[mode]
            label, color, width = MODE_STYLE[mode]
            line, = self.ax_k.plot([], [], color=color, lw=width, label=label)
            self._series.append((line, t, result.K))
            k_values.extend(result.K)

        cmap = matplotlib.colormaps['tab10']
        # Отрисовка u(t) и N(t) строго для RL (чтобы не было визуального шума)
        if 'rl' in results:
            u = results['rl'].u
            for j, label in enumerate(zone_labels()):
                line, = self.ax_u.plot([], [], color=cmap(j), lw=2.5, label=label)
                self._series.append((line, t, u[:, j]))
        self.ax_u.axhline(IDLE_THRESHOLD, color='orange', ls='--', lw=1.8, label=f"П/Г Х.Х.")
        self.ax_u.axhline(-IDLE_THRESHOLD, color='orange', ls='--', lw=1.8)

        u_values = results['rl'].u if 'rl' in results else np.zeros(1)
        u_min, u_max = u_values.min(), u_values.max()
        span_u = max(u_max - u_min, 0.5)
        self.ax_u.set_ylim(u_min - 0.1 * span_u, u_max + 0.5 * span_u)

        reference = results.get('rl', next(iter(results.values())))
        n_values_all = []

        for i, (symbol, name, _) in enumerate(PARAMETERS):
            line, = self.ax_n.plot([], [], color=cmap(i), lw=2.5, label=symbol)
            self._series.append((line, t, reference.N[:, i]))
            n_values_all.extend(reference.N[:, i])

        self.ax_n.axhline(0.5, color='k', alpha=0.3, lw=1.5)
        self.ax_n.axhline(N_LOW, color='k', ls='--', alpha=0.4, lw=1.5)
        self.ax_n.axhline(N_HIGH, color='k', ls='--', alpha=0.4, lw=1.5)

        n_values_all = np.array(n_values_all)
        n_min, n_max = n_values_all.min(), n_values_all.max()
        span_n = max(n_max - n_min, 0.15)
        self.ax_n.set_ylim(n_min - 0.45 * span_n, n_max + 0.1 * span_n)

        k_min, k_max = min(k_values), max(k_values)
        span_k = max(k_max - k_min, 0.1)
        self.ax_k.set_ylim(k_min - 0.05 * span_k, k_max + 0.45 * span_k)

        self.ax_k.set_xlim(0, T_TOTAL - 1)

        self.ax_k.set_title(f"{scenario}: {description}  (масштаб {scale:.2f}, seed {seed})", fontsize=16, pad=10)
        self.ax_k.set_ylabel("K(t)", fontsize=14)
        self.ax_u.set_ylabel("u(t)", fontsize=14)
        self.ax_n.set_ylabel("Nᵢ(t)", fontsize=14)
        self.ax_n.set_xlabel("Такт моделирования t", fontsize=14)

        for ax in (self.ax_k, self.ax_u, self.ax_n):
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=12)

        self.ax_k.legend(loc='upper right', ncol=2, fontsize=12, framealpha=0.9)
        self.ax_u.legend(loc='upper right', ncol=5, fontsize=12, framealpha=0.9)
        self.ax_n.legend(loc='lower right', ncol=4, fontsize=12, framealpha=0.9)

        self._heads = []
        for line, _, _ in self._series:
            head, = line.axes.plot([], [], 'o', color=line.get_color(), ms=9)
            self._heads.append(head)

        self._blit.artists = [line for line, _, _ in self._series] + self._heads
        for artist in self._blit.artists:
            artist.set_animated(True)
            artist.set_data([], [])
        self.canvas.draw()

    def set_frame(self, count):
        if not self._series:
            return
        for (line, t, y), head in zip(self._series, self._heads):
            line.set_data(t[:count], y[:count])
            if 0 < count < T_TOTAL:
                head.set_data([t[count - 1]], [y[count - 1]])
            else:
                head.set_data([], [])
        self._blit.update()


class ChartPlot(FigureWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax_x, self.ax_mr = self.figure.subplots(
            2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 2]}
        )
        self.ax_x.set_title("Выполните моделирование на вкладке «Моделирование»")
        self.canvas.draw_idle()

    def show_chart(self, result, index, mode_label):
        self.ax_x.clear()
        self.ax_mr.clear()
        t = np.arange(T_TOTAL)
        x = result.X[:, index]
        cl = plant.x_bar[index]
        ucl = plant.UCL[index]
        lcl = plant.LCL[index]

        outside = (x > ucl) | (x < lcl)
        self.ax_x.plot(t, x, color='tab:blue', marker='o', ms=5, lw=2.0)
        self.ax_x.scatter(t[outside], x[outside], color='red', zorder=3, s=45,
                          label="Вне контрольных границ")
        self.ax_x.axhline(cl, color='green', label=f"CL = {cl:.3f}")
        self.ax_x.axhline(ucl, color='red', ls='--', label=f"UCL = {ucl:.3f}")
        self.ax_x.axhline(lcl, color='red', ls='--', label=f"LCL = {lcl:.3f}")

        mr = np.abs(np.diff(x))
        mr_bar = (ucl - cl) / A2
        ucl_mr = D4 * mr_bar
        self.ax_mr.plot(t[1:], mr, color='tab:purple', marker='o', ms=5, lw=2.0)
        self.ax_mr.axhline(mr_bar, color='green', label=f"MR̄ = {mr_bar:.3f}")
        self.ax_mr.axhline(ucl_mr, color='red', ls='--', label=f"UCL = {ucl_mr:.3f}")

        for ax in (self.ax_x, self.ax_mr):
            ax.axvline(T_START, color='gray', ls=':')
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=12)
            ax.legend(loc='upper left', fontsize=12)

        symbol, name, operation = PARAMETERS[index]
        self.ax_x.set_title(f"Карта X–MR: {symbol} — {name} ({operation}); {mode_label}", fontsize=16, pad=10)
        self.ax_x.set_ylabel(symbol, fontsize=14)
        self.ax_mr.set_ylabel("MR", fontsize=14)
        self.ax_mr.set_xlabel("Такт моделирования t", fontsize=14)
        self.ax_x.set_xlim(0, T_TOTAL - 1)
        fit_ylim(self.ax_x, x, include=(cl,), pad_hi=0.1)
        fit_ylim(self.ax_mr, mr, include=(mr_bar,), pad_hi=0.1)
        self.canvas.draw_idle()


class TrainingPlot(FigureWidget):
    SMOOTH = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ax_r, self.ax_j = self.figure.subplots(2, 1, sharex=True)
        self.reset()

    def reset(self):
        self._it, self._ret = [], []
        self._jit, self._j = [], []
        for ax in (self.ax_r, self.ax_j):
            ax.clear()
            ax.grid(alpha=0.3, which='both')
            ax.tick_params(labelsize=12)
        self.ax_r.set_title("Кривые обучения (PPO)", fontsize=16)
        self.ax_r.set_ylabel("Доход за эпизод", fontsize=14)
        self.ax_j.set_ylabel("Метрика J (проверка)", fontsize=14)
        self.ax_j.set_xlabel("Итерация", fontsize=14)
        self.ax_j.set_yscale('log')
        self._ret_line, = self.ax_r.plot([], [], color='tab:blue', alpha=0.35, lw=1.5, label='Обучающая выборка')
        self._avg_line, = self.ax_r.plot([], [], color='tab:blue', lw=2.5, label=f'Скользящее среднее ({self.SMOOTH})')
        self._j_line, = self.ax_j.plot([], [], color='tab:orange', lw=2.5, marker='o', ms=5,
                                       label='Проверочная выборка')
        self.ax_r.legend(loc='lower right', fontsize=12)
        self.ax_j.legend(loc='upper right', fontsize=12)
        self.canvas.draw_idle()

    def add_point(self, iteration, ret, val_j=None):
        self._it.append(iteration)
        self._ret.append(ret)
        self._ret_line.set_data(self._it, self._ret)
        k = min(self.SMOOTH, len(self._ret))
        avg = np.convolve(self._ret, np.ones(k) / k, mode='valid')
        self._avg_line.set_data(self._it[k - 1:], avg)
        if val_j is not None and np.isfinite(val_j) and val_j > 0:
            self._jit.append(iteration)
            self._j.append(val_j)
            self._j_line.set_data(self._jit, self._j)
        for ax in (self.ax_r, self.ax_j):
            ax.relim()
            ax.autoscale_view()
        self.canvas.draw_idle()