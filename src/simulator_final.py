import sys
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QComboBox, QMessageBox)


class Simulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интерактивный симулятор цепей RC и RLC")
        self.setGeometry(100, 100, 500, 400)

        # Тип цепи
        self.label_type = QLabel("Тип цепи:")
        self.combo_type = QComboBox()
        self.combo_type.addItems(["RC", "RLC"])

        # Сопротивление
        self.label_R = QLabel("Сопротивление R (Ом):")
        self.input_R = QLineEdit("1000")

        # Емкость
        self.label_C = QLabel("Емкость C (Ф):")
        self.input_C = QLineEdit("0.001")

        # Индуктивность (только для RLC)
        self.label_L = QLabel("Индуктивность L (Гн):")
        self.input_L = QLineEdit("0.01")

        # Входное напряжение
        self.label_Vin = QLabel("Входное напряжение V_in (В):")
        self.input_Vin = QLineEdit("1")

        # Время моделирования
        self.label_time = QLabel("Время моделирования (сек):")
        self.input_time = QLineEdit("5")

        # Кнопка запуска
        self.btn_run = QPushButton("Считать цепь")
        self.btn_run.clicked.connect(self.run_simulation)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label_type)
        layout.addWidget(self.combo_type)
        layout.addWidget(self.label_R)
        layout.addWidget(self.input_R)
        layout.addWidget(self.label_C)
        layout.addWidget(self.input_C)
        layout.addWidget(self.label_L)
        layout.addWidget(self.input_L)
        layout.addWidget(self.label_Vin)
        layout.addWidget(self.input_Vin)
        layout.addWidget(self.label_time)
        layout.addWidget(self.input_time)
        layout.addWidget(self.btn_run)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def run_simulation(self):
        try:
            R = float(self.input_R.text())
            C = float(self.input_C.text())
            Vin = float(self.input_Vin.text())
            T = float(self.input_time.text())
            dt = 0.00001
            t = np.arange(0, T, dt)
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Введите корректные числовые значения!")
            return

        # Проверка параметров
        if not (1e-6 <= C <= 1e-2):
            QMessageBox.critical(self, "Ошибка", "Емкость C должна быть в диапазоне 1e-6 ... 1e-2 Ф")
            return
        if not (1e-4 <= R <= 1e6):
            QMessageBox.critical(self, "Ошибка", "Сопротивление R должно быть в диапазоне 1e-4 ... 1e6 Ом")
            return

        circuit_type = self.combo_type.currentText()
        V_out = None
        I_out = None
        if circuit_type == "RC":
            # RC цепь: dV/dt = (Vin - V)/RC
            V_out = np.zeros_like(t)
            for i in range(1, len(t)):
                V_out[i] = V_out[i-1] + (dt/(R*C))*(Vin - V_out[i-1])
            I_out = (Vin - V_out)/R
        elif circuit_type == "RLC":
            try:
                L = float(self.input_L.text())
            except ValueError:
                QMessageBox.critical(self, "Ошибка", "Введите корректное значение L!")
                return
            if not (1e-6 <= L <= 1):
                QMessageBox.critical(self, "Ошибка", "Индуктивность L должна быть в диапазоне 1e-6 ... 1 Гн")
                return
            # RLC цепь: dI/dt = (Vin - R*I - V)/L, dV/dt = I/C
            V_out = np.zeros_like(t)
            I_out = np.zeros_like(t)
            def dIdt(I, V):
                return (Vin - R*I - V)/L
            def dVdt(I):
                return I/C
            max_abs = 1e6  # Ограничение на значения
            for i in range(1, len(t)):
                # RK4 for I
                k1_I = dIdt(I_out[i-1], V_out[i-1])
                k1_V = dVdt(I_out[i-1])

                k2_I = dIdt(I_out[i-1] + 0.5*dt*k1_I, V_out[i-1] + 0.5*dt*k1_V)
                k2_V = dVdt(I_out[i-1] + 0.5*dt*k1_I)

                k3_I = dIdt(I_out[i-1] + 0.5*dt*k2_I, V_out[i-1] + 0.5*dt*k2_V)
                k3_V = dVdt(I_out[i-1] + 0.5*dt*k2_I)

                k4_I = dIdt(I_out[i-1] + dt*k3_I, V_out[i-1] + dt*k3_V)
                k4_V = dVdt(I_out[i-1] + dt*k3_I)

                I_out[i] = I_out[i-1] + (dt/6)*(k1_I + 2*k2_I + 2*k3_I + k4_I)
                V_out[i] = V_out[i-1] + (dt/6)*(k1_V + 2*k2_V + 2*k3_V + k4_V)
                # Ограничение на значения
                if abs(I_out[i]) > max_abs:
                    I_out[i] = np.sign(I_out[i]) * max_abs
                if abs(V_out[i]) > max_abs:
                    V_out[i] = np.sign(V_out[i]) * max_abs

        # Сохраняем CSV
        import os
        data_dir = "/Users/qqwarklem/Documents/Xcode/project_simulator/data"
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, "circuit_output.txt")
        np.savetxt(
            csv_path,
            np.column_stack((t, V_out, I_out)),
            fmt="%.6f",
            delimiter="\t",
            header="time\tV_out\tI_out",
            comments=''
        )

        # Построение графиков
        plt.figure(figsize=(10,5))
        plt.plot(t, V_out, label="V_out (В)")
        plt.plot(t, I_out, label="I (A)")
        plt.xlabel("Время (с)")
        plt.ylabel("Напряжение / Ток")
        plt.title(f"{circuit_type} цепь")
        plt.grid(True)
        plt.legend()
        plt.show()

        QMessageBox.information(self, "Готово", "Симуляция завершена! Данные сохранены в data/circuit_output.txt")


# Запуск приложения
app = QApplication(sys.argv)
window = Simulator()
window.show()
sys.exit(app.exec())