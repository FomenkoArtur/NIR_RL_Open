import sys

import hw

hw.prepare()

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.tuning import add_tuning_tabs


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    add_tuning_tabs(window)                      # вкладка «Подбор ПИД»
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()