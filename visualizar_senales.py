"""
Script para visualizar las señales del LVDT y sensor de proximidad
y verificar la detección de eventos (flancos y cruces por cero)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Archivo de ejemplo para analizar
archivo = r"c:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\G12\Forzada-amortiguada4.3hz\Forzada-amortiguada4.3hz_1.csv"
frecuencia = 4.3  # Hz


def leer_csv(filepath):
    """Lee archivo CSV"""
    try:
        df = pd.read_csv(filepath, sep=';', decimal=',', skiprows=2, header=None)
        tiempo_ms = df.iloc[:, 0].values
        canal_a = df.iloc[:, 1].values    # LVDT - Señal cuadrada
        canal_b = df.iloc[:, 2].values    # Proximidad - Señal sinusoidal
        
        tiempo_s = tiempo_ms / 1000.0
        return tiempo_s, canal_a, canal_b
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


def detectar_flanco_lvdt(tiempo, canal_a):
    """Detecta flancos ascendentes en señal cuadrada"""
    umbral = (np.max(canal_a) + np.min(canal_a)) / 2
    flancos = []
    
    for i in range(len(canal_a) - 1):
        if canal_a[i] < umbral and canal_a[i+1] >= umbral:
            t_flanco = tiempo[i] + (umbral - canal_a[i]) * (tiempo[i+1] - tiempo[i]) / (canal_a[i+1] - canal_a[i])
            flancos.append(t_flanco)
    
    return flancos, umbral


def detectar_cruce_cero_proximidad(tiempo, canal_b):
    """Detecta cruces por cero ascendentes en señal sinusoidal"""
    cruces = []
    
    for i in range(len(canal_b) - 1):
        if canal_b[i] <= 0 and canal_b[i+1] > 0:
            t_cruce = tiempo[i] - canal_b[i] * (tiempo[i+1] - tiempo[i]) / (canal_b[i+1] - canal_b[i])
            cruces.append(t_cruce)
    
    return cruces


# Leer datos
print(f"Analizando archivo: {Path(archivo).name}")
print(f"Frecuencia: {frecuencia} Hz")
print(f"Período teórico: {1/frecuencia*1000:.2f} ms")
print()

tiempo, canal_a, canal_b = leer_csv(archivo)

if tiempo is not None:
    # Detectar eventos
    flancos, umbral_a = detectar_flanco_lvdt(tiempo, canal_a)
    cruces = detectar_cruce_cero_proximidad(tiempo, canal_b)
    
    print(f"Flancos LVDT detectados: {len(flancos)}")
    print(f"Cruces proximidad detectados: {len(cruces)}")
    print()
    
    if len(flancos) > 0 and len(cruces) > 0:
        # Usar el segundo de cada uno para evitar transitorios
        idx_flanco = 1 if len(flancos) > 1 else 0
        idx_cruce = 1 if len(cruces) > 1 else 0
        
        t_flanco = flancos[idx_flanco]
        t_cruce = cruces[idx_cruce]
        
        print(f"Evento seleccionado:")
        print(f"  Flanco LVDT:     {t_flanco*1000:.3f} ms")
        print(f"  Cruce proximidad: {t_cruce*1000:.3f} ms")
        print()
        
        # Calcular Δt con ambas opciones
        T = 1 / frecuencia
        
        print("="*60)
        print("OPCIÓN 1: Δt = t_flanco - t_cruce (fórmula actual)")
        delta_t_1 = t_flanco - t_cruce
        phi_1 = -(delta_t_1 / T) * 2 * np.pi * 180 / np.pi
        print(f"  Δt = {delta_t_1*1000:.3f} ms")
        print(f"  φ = -(Δt/T)×2π = {phi_1:.2f}°")
        
        print()
        print("OPCIÓN 2: Δt = t_cruce - t_flanco (invertido)")
        delta_t_2 = t_cruce - t_flanco
        phi_2 = -(delta_t_2 / T) * 2 * np.pi * 180 / np.pi
        print(f"  Δt = {delta_t_2*1000:.3f} ms")
        print(f"  φ = -(Δt/T)×2π = {phi_2:.2f}°")
        
        print()
        print("OPCIÓN 3: φ = +(Δt/T)×2π (sin negativo)")
        phi_3 = (delta_t_1 / T) * 2 * np.pi * 180 / np.pi
        print(f"  Δt = {delta_t_1*1000:.3f} ms")
        print(f"  φ = +(Δt/T)×2π = {phi_3:.2f}°")
        print("="*60)
        print()
        
        # Crear gráficas
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        
        # Limitar a los primeros 1.5 segundos para ver mejor
        mask = tiempo <= 1.5
        t_plot = tiempo[mask]
        
        # Gráfica 1: Canal A (LVDT - Cuadrada)
        ax1.plot(t_plot*1000, canal_a[mask], 'b-', linewidth=1, label='Canal A (LVDT)')
        ax1.axhline(umbral_a, color='gray', linestyle='--', alpha=0.5, label=f'Umbral = {umbral_a:.2f}V')
        
        # Marcar todos los flancos en el rango visible
        for f in flancos:
            if f <= 1.5:
                ax1.axvline(f*1000, color='red', linestyle='--', alpha=0.7)
        
        # Destacar el flanco usado
        ax1.axvline(t_flanco*1000, color='red', linewidth=2, label=f'Flanco usado: {t_flanco*1000:.2f}ms')
        ax1.set_xlabel('Tiempo (ms)')
        ax1.set_ylabel('Voltaje (V)')
        ax1.set_title('Canal A: LVDT (Señal Cuadrada) - Detección de Flancos Ascendentes')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfica 2: Canal B (Proximidad - Sinusoidal)
        ax2.plot(t_plot*1000, canal_b[mask], 'g-', linewidth=1, label='Canal B (Proximidad)')
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Cero')
        
        # Marcar todos los cruces en el rango visible
        for c in cruces:
            if c <= 1.5:
                ax2.axvline(c*1000, color='orange', linestyle='--', alpha=0.7)
        
        # Destacar el cruce usado
        ax2.axvline(t_cruce*1000, color='orange', linewidth=2, label=f'Cruce usado: {t_cruce*1000:.2f}ms')
        ax2.set_xlabel('Tiempo (ms)')
        ax2.set_ylabel('Voltaje (mV)')
        ax2.set_title('Canal B: Proximidad (Señal Sinusoidal) - Detección de Cruces por Cero Ascendentes')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Gráfica 3: Ambas señales superpuestas (normalizadas)
        # Normalizar para comparar fases
        canal_a_norm = (canal_a[mask] - np.min(canal_a[mask])) / (np.max(canal_a[mask]) - np.min(canal_a[mask]))
        canal_b_norm = (canal_b[mask] - np.min(canal_b[mask])) / (np.max(canal_b[mask]) - np.min(canal_b[mask]))
        
        ax3.plot(t_plot*1000, canal_a_norm, 'b-', linewidth=1.5, label='Canal A (LVDT) normalizado', alpha=0.7)
        ax3.plot(t_plot*1000, canal_b_norm, 'g-', linewidth=1.5, label='Canal B (Proximidad) normalizado', alpha=0.7)
        ax3.axvline(t_flanco*1000, color='red', linewidth=2, linestyle='--', label=f'Flanco LVDT: {t_flanco*1000:.2f}ms')
        ax3.axvline(t_cruce*1000, color='orange', linewidth=2, linestyle='--', label=f'Cruce Prox: {t_cruce*1000:.2f}ms')
        
        # Mostrar Δt
        y_pos = 0.5
        ax3.annotate('', xy=(t_cruce*1000, y_pos), xytext=(t_flanco*1000, y_pos),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax3.text((t_flanco*1000 + t_cruce*1000)/2, y_pos + 0.05, 
                f'Δt = {abs(delta_t_1)*1000:.2f}ms', 
                ha='center', fontsize=10, color='purple', fontweight='bold')
        
        ax3.set_xlabel('Tiempo (ms)')
        ax3.set_ylabel('Señal normalizada')
        ax3.set_title(f'Comparación de Fase: LVDT vs Proximidad (Δt = {delta_t_1*1000:.3f}ms)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        
        # Guardar figura
        output_file = Path(archivo).parent.parent / "visualizacion_senales_phi.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfica guardada en: {output_file}")
        
        plt.show()

else:
    print("No se pudo leer el archivo")
