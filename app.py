from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from scipy import signal as scipy_signal
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATES_DIR)

class SignalGenerator:
    def __init__(self):
        self.signal_type = 'sine'
        self.frequency = 1000
        self.amplitude = 1.0
        self.offset = 0.0
        self.phase = 0.0
        
    def generate(self, time_array):
        t = time_array
        if self.signal_type == 'sine':
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase) + self.offset
        elif self.signal_type == 'square':
            return self.amplitude * scipy_signal.square(2 * np.pi * self.frequency * t + self.phase) + self.offset
        elif self.signal_type == 'sawtooth':
            return self.amplitude * scipy_signal.sawtooth(2 * np.pi * self.frequency * t + self.phase) + self.offset
        elif self.signal_type == 'triangle':
            return self.amplitude * scipy_signal.sawtooth(2 * np.pi * self.frequency * t + self.phase, 0.5) + self.offset
        elif self.signal_type == 'impulse':
            signal = np.zeros_like(t)
            signal[len(t)//2] = self.amplitude
            return signal
        elif self.signal_type == 'noise':
            return self.amplitude * np.random.normal(0, 1, len(t)) + self.offset
        return np.zeros_like(t)

class CircuitTopology:
    def __init__(self, components):
        self.components = components
        self.analyze_topology()
    
    def analyze_topology(self):
        """Анализирует реальную топологию цепи с учетом соединений"""
        self.series_branch = []
        self.parallel_branches = []
        self.voltage_sources = []
        
        # Сортируем компоненты по положению (координате x)
        sorted_components = sorted(self.components, key=lambda x: x.get('x', 0))
        
        current_branch = []
        for comp in sorted_components:
            if comp['type'] == 'voltage-source':
                self.voltage_sources.append(comp)
                continue
                
            if comp['connection'] == 'series':
                if current_branch:
                    self.parallel_branches.append(current_branch)
                    current_branch = []
                self.series_branch.append(comp)
            elif comp['connection'] == 'parallel':
                current_branch.append(comp)
        
        if current_branch:
            self.parallel_branches.append(current_branch)
    
    def calculate_impedance(self, frequency):
        """Расчет полного комплексного импеданса с учетом фазы"""
        w = 2 * np.pi * frequency
        
        # Импеданс последовательной ветви
        Z_series = 0
        for comp in self.series_branch:
            Z_series += self._component_impedance(comp, w)
        
        # Импеданс параллельных ветвей
        Y_parallel = 0
        for branch in self.parallel_branches:
            Z_branch = 0
            for comp in branch:
                Z_branch += self._component_impedance(comp, w)
            if Z_branch != 0:
                Y_parallel += 1 / Z_branch
        
        Z_parallel = 1 / Y_parallel if Y_parallel != 0 else 0
        
        return Z_series + Z_parallel
    
    def _component_impedance(self, comp, w):
        """Импеданс отдельного компонента"""
        if comp['type'] == 'resistor':
            return comp['value']
        elif comp['type'] == 'capacitor':
            return 1 / (1j * w * comp['value']) if comp['value'] > 0 else 1e12
        elif comp['type'] == 'inductor':
            return 1j * w * comp['value']
        return 0
    
    def get_circuit_type(self):
        """Определяет тип цепи на основе компонентов"""
        has_r = any(c['type'] == 'resistor' for c in self.components)
        has_c = any(c['type'] == 'capacitor' for c in self.components)
        has_l = any(c['type'] == 'inductor' for c in self.components)
        
        if has_r and has_c and not has_l:
            return 'RC'
        elif has_r and has_l and not has_c:
            return 'RL'
        elif has_r and has_c and has_l:
            return 'RLC'
        elif has_r:
            return 'Resistive'
        else:
            return 'Simple'
    
    def get_circuit_parameters(self):
        """Возвращает параметры цепи для отображения"""
        R, L, C = self._extract_rlc_parameters()
        Vsrc = self.voltage_sources[0]['value'] if self.voltage_sources else 1.0
        
        return {
            "R": R,
            "L": L, 
            "C": C,
            "Vsrc": Vsrc
        }
    
    def _extract_rlc_parameters(self):
        """Извлекает параметры R, L, C с учетом топологии"""
        R, L, C = 1000, 0.001, 1e-6  # значения по умолчанию
        
        # Сбрасываем для перерасчета
        R, L, C = 0, 0, 0
        
        # Последовательные компоненты
        for comp in self.series_branch:
            if comp['type'] == 'resistor':
                R += comp['value']
            elif comp['type'] == 'inductor':
                L += comp['value']
            elif comp['type'] == 'capacitor':
                if C == 0:
                    C = comp['value']
                else:
                    C = 1 / (1/C + 1/comp['value'])  # последовательные конденсаторы
        
        # Параллельные ветви
        for branch in self.parallel_branches:
            R_branch, L_branch, C_branch = 0, 0, 0
            for comp in branch:
                if comp['type'] == 'resistor':
                    R_branch += comp['value']
                elif comp['type'] == 'inductor':
                    L_branch += comp['value']
                elif comp['type'] == 'capacitor':
                    C_branch += comp['value']
            
            # Параллельное соединение резисторов
            if R_branch > 0:
                if R == 0:
                    R = R_branch
                else:
                    R = 1 / (1/R + 1/R_branch)
            
            # Параллельное соединение катушек  
            if L_branch > 0:
                if L == 0:
                    L = L_branch
                else:
                    L = 1 / (1/L + 1/L_branch)
            
            # Параллельное соединение конденсаторов
            C += C_branch
        
        # Устанавливаем значения по умолчанию если нули
        if R <= 0: R = 1000
        if L <= 0: L = 0.001
        if C <= 0: C = 1e-6
        
        return R, L, C

class CircuitSimulatorBackend:
    def __init__(self):
        self.signal_generator = SignalGenerator()
    
    def simulate_circuit_response(self, components, input_signal, time_array):
        """Симуляция отклика реальной цепи на входной сигнал"""
        if not components:
            return input_signal, np.zeros_like(input_signal)
        
        topology = CircuitTopology(components)
        circuit_type = topology.get_circuit_type()
        
        # Для разных типов цепей используем разные модели
        if circuit_type == 'RC':
            return self._simulate_rc_response(topology, input_signal, time_array)
        elif circuit_type == 'RLC':
            return self._simulate_rlc_response(topology, input_signal, time_array)
        elif circuit_type == 'RL':
            return self._simulate_rl_response(topology, input_signal, time_array)
        else:
            # Для резистивных цепей - просто ослабление
            Z = topology.calculate_impedance(1000)
            gain = min(1.0, 1000 / np.abs(Z)) if np.abs(Z) > 0 else 1.0
            return input_signal, input_signal * gain
    
    def _simulate_rc_response(self, topology, input_signal, t):
        """Симуляция RC-цепи с учетом реальной топологии"""
        R, L, C = topology._extract_rlc_parameters()
        
        if R <= 0 or C <= 0:
            return input_signal, input_signal
        
        # Решаем диффур: dVc/dt = (Vin - Vc) / (R*C)
        def rc_equation(Vc, t, Vin_func):
            return (Vin_func(t) - Vc) / (R * C)
        
        Vin_interp = interp1d(t, input_signal, kind='linear', fill_value="extrapolate")
        
        # Решаем ОДУ
        Vc_initial = 0
        Vc = odeint(rc_equation, Vc_initial, t, args=(Vin_interp,))
        
        output_signal = Vc.flatten()
        current_signal = (input_signal - output_signal) / R
        
        return output_signal, current_signal
    
    def _simulate_rlc_response(self, topology, input_signal, t):
        """Симуляция RLC цепи"""
        R, L, C = topology._extract_rlc_parameters()
        
        if R <= 0 or L <= 0 or C <= 0:
            return input_signal, input_signal
        
        # Система уравнений для RLC
        def rlc_equations(state, t, Vin_func):
            I, Vc = state
            dI_dt = (Vin_func(t) - R * I - Vc) / L
            dVc_dt = I / C
            return [dI_dt, dVc_dt]
        
        Vin_interp = interp1d(t, input_signal, kind='linear', fill_value="extrapolate")
        
        # Начальные условия
        state0 = [0, 0]
        states = odeint(rlc_equations, state0, t, args=(Vin_interp,))
        
        I_out = states[:, 0]
        Vc_out = states[:, 1]
        
        return Vc_out, I_out
    
    def _simulate_rl_response(self, topology, input_signal, t):
        """Симуляция RL цепи"""
        R, L, C = topology._extract_rlc_parameters()
        
        if R <= 0 or L <= 0:
            return input_signal, input_signal
        
        # Уравнение для RL: dI/dt = (Vin - R*I) / L
        def rl_equation(I, t, Vin_func):
            return (Vin_func(t) - R * I) / L
        
        Vin_interp = interp1d(t, input_signal, kind='linear', fill_value="extrapolate")
        
        I_initial = 0
        I_out = odeint(rl_equation, I_initial, t, args=(Vin_interp,))
        
        output_signal = I_out.flatten() * R  # Напряжение на резисторе
        current_signal = I_out.flatten()
        
        return output_signal, current_signal
    
    def frequency_analysis(self, components):
        """Полный АЧХ и ФЧХ анализ с учетом фазы"""
        topology = CircuitTopology(components)
        frequencies = np.logspace(0, 6, 500)
        
        magnitude = np.zeros_like(frequencies)
        phase = np.zeros_like(frequencies)
        impedance_magnitude = np.zeros_like(frequencies)
        
        for i, f in enumerate(frequencies):
            Z = topology.calculate_impedance(f)
            
            # Упрощенная передаточная функция
            H = 1 / (1 + Z/1000)
            
            magnitude[i] = 20 * np.log10(np.abs(H)) if np.abs(H) > 1e-12 else -120
            phase[i] = np.angle(H, deg=True)
            impedance_magnitude[i] = np.abs(Z)
        
        return frequencies, magnitude, phase, impedance_magnitude

    def plot_to_base64(self, t, V_out, I_out, circuit_type):
        """Создание графика и преобразование в base64"""
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(t, V_out, 'b-', linewidth=2, label="Напряжение (В)")
        plt.xlabel("Время (с)")
        plt.ylabel("Напряжение (В)")
        plt.title(f"Напряжение - {circuit_type}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, I_out * 1000, 'r-', linewidth=2, label="Ток (мА)")
        plt.xlabel("Время (с)")
        plt.ylabel("Ток (мА)")
        plt.title(f"Ток - {circuit_type}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
    
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

simulator = CircuitSimulatorBackend()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        components = data.get('components', [])
        simulation_type = data.get('simulation_type', 'rc')
        
        if not components:
            return jsonify({"status": "error", "message": "Нет компонентов для симуляции"})
        
        # Создаем топологию для получения параметров
        topology = CircuitTopology(components)
        parameters = topology.get_circuit_parameters()
        
        if simulation_type == 'frequency':
            # АЧХ анализ с учетом фазы
            f, mag, phase, impedance = simulator.frequency_analysis(components)
            return jsonify({
                "status": "success",
                "type": "frequency",
                "frequencies": f.tolist(),
                "magnitude": mag.tolist(),
                "phase": phase.tolist(),
                "impedance": impedance.tolist(),
                "parameters": parameters  # ✅ ДОБАВЛЕНО!
            })
        else:
            # Временной анализ
            t = np.linspace(0, 0.1, 1000)
            
            # Генерируем входной сигнал в зависимости от типа симуляции
            if simulation_type == 'impulse':
                input_signal = np.zeros_like(t)
                input_signal[100] = parameters['Vsrc']  # Импульс
            elif simulation_type == 'noise':
                input_signal = parameters['Vsrc'] + 0.1 * parameters['Vsrc'] * np.random.normal(0, 1, len(t))
            else:
                # Синус для обычной симуляции
                input_signal = parameters['Vsrc'] * np.sin(2 * np.pi * 100 * t)
            
            # Симулируем отклик реальной цепи
            output_signal, current_signal = simulator.simulate_circuit_response(components, input_signal, t)
            
            circuit_type = topology.get_circuit_type()
            
            image_base64 = simulator.plot_to_base64(t, output_signal, current_signal, circuit_type)
            
            return jsonify({
                "status": "success",
                "type": "time_domain",
                "image": image_base64,
                "time": t.tolist(),
                "voltage": output_signal.tolist(),
                "current": current_signal.tolist(),
                "input_signal": input_signal.tolist(),
                "parameters": parameters  # ✅ ДОБАВЛЕНО!
            })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Ошибка симуляции: {str(e)}"})

@app.route('/api/oscilloscope', methods=['POST'])
def oscilloscope():
    try:
        data = request.json
        signal_type = data.get('signal_type', 'sine')
        frequency = data.get('frequency', 1000)
        amplitude = data.get('amplitude', 1.0)
        components = data.get('components', [])
        
        # Генерируем входной сигнал
        simulator.signal_generator.signal_type = signal_type
        simulator.signal_generator.frequency = frequency
        simulator.signal_generator.amplitude = amplitude
        
        t = np.linspace(0, 0.01, 1000)
        input_signal = simulator.signal_generator.generate(t)
        
        # Пропускаем через реальную цепь!
        output_signal, current_signal = simulator.simulate_circuit_response(components, input_signal, t)
        
        return jsonify({
            "status": "success",
            "time": t.tolist(),
            "input_signal": input_signal.tolist(),
            "output_signal": output_signal.tolist(),
            "current_signal": current_signal.tolist()
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/save_circuit', methods=['POST'])
def save_circuit():
    try:
        data = request.json
        components = data.get('components', [])
        filename = data.get('filename', 'circuit.json')
        
        if not components:
            return jsonify({"status": "error", "message": "Нет схемы для сохранения"})
        
        filepath = os.path.join('saved_circuits', filename)
        with open(filepath, 'w') as f:
            json.dump(components, f, indent=2)
            
        return jsonify({"status": "success", "message": f"Схема сохранена как {filename}"})
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Ошибка сохранения: {str(e)}"})

if __name__ == '__main__':
    os.makedirs('saved_circuits', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Запуск сервера...")
    print("Откройте в браузере: http://localhost:5000")
    app.run(debug=False, port=8000, host='0.0.0.0')