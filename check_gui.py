import sys
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # запускаем без реального окна

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.tuning import add_tuning_tabs

def main():
    print("=== ЗАПУСК ПРОВЕРКИ GUI ===")
    app = QApplication(sys.argv)
    window = MainWindow()
    add_tuning_tabs(window)  # добавляем подбор ПИД
    
    # 1. Проверка вкладок
    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    print(f"Найдено вкладок: {len(tabs)}")
    for t in tabs:
        print(f"  - {t}")
        
    print("\n=== ПРОВЕРКА ПРАВИЛ ===")
    
    # 2. Проверка наличия нужных вкладок
    required_tabs = ["Моделирование", "Контрольные карты Шухарта", "Обучение регулятора", 
                     "Таблица 1 (эксперимент)", "Параметры модели", "АФХ", "Боде", "Подбор ПИД"]
    for req in required_tabs:
        found = any(req in t for t in tabs)
        print(f"Вкладка '{req}': {'OK' if found else 'ОШИБКА (не найдена)'}")
        
    # 3. Проверка отсутствия лишних вкладок
    banned_tabs = ["Подбор архитектуры"]
    for ban in banned_tabs:
        found = any(ban in t for t in tabs)
        print(f"Вкладка '{ban}' (должна отсутствовать): {'ОШИБКА (присутствует)' if found else 'OK (удалена)'}")
        
    # 4. Проверка ComboBox номенклатуры
    from PySide6.QtWidgets import QComboBox
    combos = window.findChildren(QComboBox)
    nomenclature_found = False
    for c in combos:
        items = [c.itemText(i) for i in range(c.count())]
        if "Вал" in items or "Фланец" in items:
            nomenclature_found = True
            print(f"ComboBox номенклатуры найден! Элементы: {items}")
            break
    if not nomenclature_found:
        print("ComboBox номенклатуры: ОШИБКА (не найден)")

    # 5. Проверка наличия PID/MPC (косвенно по импортам)
    import gui.main_window as m
    print(f"Импорт PIDController: {'OK' if hasattr(m, 'PIDController') else 'ОШИБКА'}")
    print(f"Импорт MPCController: {'OK' if hasattr(m, 'MPCController') else 'ОШИБКА'}")

    print("\n=== ПРОВЕРКА ЗАВЕРШЕНА ===")
    window.close()
    sys.exit(0)

if __name__ == "__main__":
    main()
