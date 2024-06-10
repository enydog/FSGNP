import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Cargar el archivo CSV
file_path = 'HIMdata1.csv'
data = pd.read_csv(file_path, delimiter=';')

# Suavizado con una ventana de 30 muestras
data['w_smooth'] = data['w'].rolling(window=30).mean()

# Función para calcular la varianza en una ventana deslizante
def rolling_variance(series, window=30):
    return series.rolling(window=window).var()

# Función para seleccionar el orden del polinomio en función de la varianza
def select_polyorder(variance):
    if variance > 2.0:  # Umbral para varianza alta
        return 5
    elif variance > 1.0:  # Umbral para varianza moderada
        return 3
    else:
        return 1

# Función para ajustar dinámicamente el orden del polinomio del filtro Savitzky-Golay
def adaptive_savgol_filter(series, window_length=31):
    variances = rolling_variance(series, window=window_length)
    polyorders = variances.apply(select_polyorder).fillna(1).astype(int)
    
    filtered_series = np.zeros_like(series)
    half_window = window_length // 2
    
    for i in range(half_window, len(series) - half_window):
        local_polyorder = polyorders.iloc[i]
        filtered_series[i] = savgol_filter(series[i - half_window:i + half_window + 1], window_length, local_polyorder)[half_window]
    
    return filtered_series

# Aplicar el filtro Savitzky-Golay adaptativo
data['w_savgol_adaptive'] = adaptive_savgol_filter(data['w'], window_length=31)

# Función para calcular la potencia normalizada utilizando la media móvil
def normalized_power(power_series, window=30):
    rolling_mean = power_series.rolling(window=window).mean()
    power_4th = rolling_mean ** 4
    np_power = (power_4th.mean()) ** 0.25
    return np_power

# Calcular la potencia normalizada con la media móvil
np_power = normalized_power(data['w'])

# Función para calcular la potencia normalizada utilizando el filtro Savitzky-Golay adaptativo
def normalized_power_savgol_adaptive(power_series):
    power_savgol_adaptive = adaptive_savgol_filter(power_series, window_length=31)
    power_4th = power_savgol_adaptive ** 4
    np_power_savgol_adaptive = (power_4th.mean()) ** 0.25
    return np_power_savgol_adaptive

# Calcular la potencia normalizada con el suavizado Savitzky-Golay adaptativo
np_power_savgol_adaptive = normalized_power_savgol_adaptive(data['w'])

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(data['s'], data['w'], label='Datos originales')
plt.plot(data['s'], data['w_smooth'], label='Suavizado ventana 30 muestras', linestyle='--')
plt.plot(data['s'], data['w_savgol_adaptive'], label='Filtro Savitzky-Golay Adaptativo', linestyle='-.')
plt.axhline(np_power, color='red', linestyle='-', label=f'Potencia Normalizada (Media Móvil): {np_power:.2f} W')
plt.axhline(np_power_savgol_adaptive, color='purple', linestyle='--', label=f'Potencia Normalizada (Savitzky-Golay Adaptativo): {np_power_savgol_adaptive:.2f} W')
plt.xlabel('Tiempo (s)')
plt.ylabel('Watts')
plt.title('Datos Originales, Suavizado, Filtro Savitzky-Golay Adaptativo y Potencia Normalizada')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar valores de potencia normalizada
print(f'Potencia Normalizada (Media Móvil): {np_power:.2f} W')
print(f'Potencia Normalizada (Savitzky-Golay Adaptativo): {np_power_savgol_adaptive:.2f} W')
