"""
Cálculo mejorado del ángulo de desfase φ
Métodos:
1. Tiempo entre máximos (pico a pico)
2. Correlación cruzada
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d

# Configuración
base_path = Path(r"c:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\G12")

configuraciones = {
    'Forzada-amortiguada4.2hz': {'freq': 4.2, 'tipo': 'cerrada'},
    'Forzada-amortiguada4.3hz': {'freq': 4.3, 'tipo': 'cerrada'},
    'Forzada-amortiguada4.45hz': {'freq': 4.45, 'tipo': 'cerrada'},
    'Forzada-amortiguada-abierta4.2hz': {'freq': 4.2, 'tipo': 'abierta'},
    'Forzada-amortiguada-abierta4.3hz': {'freq': 4.3, 'tipo': 'abierta'},
    'Forzada-amortiguada-abierta4.45hz': {'freq': 4.45, 'tipo': 'abierta'},
}


def leer_csv(filepath):
    """Lee archivo CSV
    Convención de canales (según observación de datos y usuario):
    - Canal A (columna 1): Proximidad -> SEÑAL CUADRADA (referencia de excitación)
    - Canal B (columna 2): LVDT       -> SEÑAL SINUSOIDAL (respuesta)
    """
    try:
        df = pd.read_csv(filepath, sep=';', decimal=',', skiprows=2, header=None)
        tiempo_ms = df.iloc[:, 0].values
        canal_prox = df.iloc[:, 1].values   # Proximidad (cuadrada)
        canal_lvdt = df.iloc[:, 2].values   # LVDT (sinusoidal)
        
        tiempo_s = tiempo_ms / 1000.0
        return tiempo_s, canal_lvdt, canal_prox
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return None, None, None


def detectar_maximos(tiempo, senal, distancia_minima=None):
    """
    Detecta los máximos de una señal
    """
    if distancia_minima is None:
        # Estimar distancia mínima basada en la frecuencia esperada
        dt = np.mean(np.diff(tiempo))
        distancia_minima = int(0.15 / dt)  # ~150ms entre picos
    
    picos, _ = signal.find_peaks(senal, distance=distancia_minima)
    
    if len(picos) > 0:
        return tiempo[picos], senal[picos]
    return np.array([]), np.array([])


def calcular_phi_maximos(tiempo, canal_lvdt, canal_prox, frecuencia):
    """
    Calcula φ midiendo el tiempo entre máximos de ambas señales
    
    Interpretación:
    - Máximo de proximidad = máximo de la excitación (fuerza)
    - Máximo de LVDT = máximo de la respuesta (desplazamiento)
    """
    # Detectar máximos
    t_max_prox, _ = detectar_maximos(tiempo, canal_prox)
    t_max_lvdt, _ = detectar_maximos(tiempo, canal_lvdt)
    
    if len(t_max_prox) < 2 or len(t_max_lvdt) < 2:
        return None
    
    # Usar los máximos del medio (evitar transitorios)
    idx_prox = len(t_max_prox) // 2
    idx_lvdt = len(t_max_lvdt) // 2
    
    # Asegurarse de tomar máximos cercanos en el mismo ciclo
    # Buscar el máximo de LVDT más cercano al máximo de proximidad
    t_ref_prox = t_max_prox[idx_prox]
    diferencias = np.abs(t_max_lvdt - t_ref_prox)
    idx_lvdt_cercano = np.argmin(diferencias)
    
    t_ref_lvdt = t_max_lvdt[idx_lvdt_cercano]
    
    # Δt = tiempo_respuesta - tiempo_excitacion
    delta_t = t_ref_lvdt - t_ref_prox
    
    # Período
    T = 1 / frecuencia
    
    # Normalizar Δt al rango [-T/2, T/2]
    while delta_t > T/2:
        delta_t -= T
    while delta_t < -T/2:
        delta_t += T
    
    # Calcular φ en radianes y grados
    # φ = (Δt/T) × 2π
    phi_rad = (delta_t / T) * 2 * np.pi
    phi_deg = phi_rad * 180 / np.pi
    
    return {
        't_max_prox': t_ref_prox,
        't_max_lvdt': t_ref_lvdt,
        'delta_t': delta_t,
        'T': T,
        'phi_rad': phi_rad,
        'phi_deg': phi_deg,
        'metodo': 'maximos'
    }


def detectar_flancos_rising(tiempo, senal):
    """Detecta flancos ascendentes en una señal (p.ej., cuadrada de proximidad)
    Retorna lista de tiempos de flanco usando interpolación lineal a un umbral medio.
    """
    umbral = (np.max(senal) + np.min(senal)) / 2.0
    flancos = []
    for i in range(len(senal) - 1):
        if senal[i] < umbral and senal[i+1] >= umbral:
            # Interpolación lineal
            t = tiempo[i] + (umbral - senal[i]) * (tiempo[i+1] - tiempo[i]) / (senal[i+1] - senal[i])
            flancos.append(t)
    return np.array(flancos), umbral


def calcular_phi_pulso_vs_maximo(tiempo, canal_lvdt, canal_prox, frecuencia):
    """Mide φ como tiempo entre el pulso de proximidad (flanco ascendente)
    y el máximo siguiente del LVDT. Δt = t_max(LVDT) - t_pulso(Prox).
    φ = (Δt/T)×360° con plegado a [-180, 180].
    """
    # Detectar eventos
    t_flancos, _ = detectar_flancos_rising(tiempo, canal_prox)
    t_max_lvdt, _ = detectar_maximos(tiempo, canal_lvdt)

    if len(t_flancos) < 2 or len(t_max_lvdt) < 2:
        return None

    T = 1.0 / frecuencia
    deltas = []
    pares = 0

    for t_pulso in t_flancos:
        # Buscar el máximo de LVDT posterior más cercano al pulso
        idx = np.searchsorted(t_max_lvdt, t_pulso, side='right')
        if idx < len(t_max_lvdt):
            t_max = t_max_lvdt[idx]
            delta_t = t_max - t_pulso
            # Si por alguna razón cayó casi un periodo después, normalizar
            while delta_t > T/2:
                delta_t -= T
            while delta_t < -T/2:
                delta_t += T
            deltas.append(delta_t)
            pares += 1

    if pares == 0:
        return None

    delta_t_med = float(np.median(deltas))
    phi_deg = (delta_t_med / T) * 360.0
    # Normalizar φ a [-180, 180]
    while phi_deg > 180.0:
        phi_deg -= 360.0
    while phi_deg < -180.0:
        phi_deg += 360.0
    phi_rad = np.deg2rad(phi_deg)

    return {
        'delta_t': delta_t_med,
        'T': T,
        'phi_rad': phi_rad,
        'phi_deg': phi_deg,
        'pares_usados': pares,
        'metodo': 'pulso_vs_maximo'
    }


def calcular_phi_correlacion(tiempo, canal_lvdt, canal_prox, frecuencia):
    """
    Calcula φ usando correlación cruzada
    Encuentra el desplazamiento temporal que maximiza la correlación
    """
    # Remover tendencia y normalizar
    lvdt_detrend = signal.detrend(canal_lvdt)
    prox_detrend = signal.detrend(canal_prox)
    
    lvdt_norm = lvdt_detrend / np.std(lvdt_detrend)
    prox_norm = prox_detrend / np.std(prox_detrend)
    
    # Correlación cruzada
    correlacion = signal.correlate(lvdt_norm, prox_norm, mode='full')
    lags = signal.correlation_lags(len(lvdt_norm), len(prox_norm), mode='full')
    
    # Convertir lags a tiempo
    dt = np.mean(np.diff(tiempo))
    lags_tiempo = lags * dt
    
    # Encontrar el lag que maximiza la correlación
    idx_max = np.argmax(correlacion)
    delta_t = lags_tiempo[idx_max]
    
    # Período
    T = 1 / frecuencia
    
    # Normalizar Δt al rango [-T/2, T/2]
    while delta_t > T/2:
        delta_t -= T
    while delta_t < -T/2:
        delta_t += T
    
    # Calcular φ
    phi_rad = (delta_t / T) * 2 * np.pi
    phi_deg = phi_rad * 180 / np.pi
    
    return {
        'delta_t': delta_t,
        'T': T,
        'phi_rad': phi_rad,
        'phi_deg': phi_deg,
        'correlacion_max': correlacion[idx_max],
        'metodo': 'correlacion'
    }


def detectar_cruces_cero_asc(tiempo, senal):
    """Detecta cruces por cero ascendentes (de negativo a positivo) en una señal
    continua (p.ej., LVDT). Retorna los tiempos interpolados de cruce.
    """
    cruces = []
    for i in range(len(senal) - 1):
        if senal[i] <= 0 and senal[i+1] > 0:
            t = tiempo[i] - senal[i] * (tiempo[i+1] - tiempo[i]) / (senal[i+1] - senal[i])
            cruces.append(t)
    return np.array(cruces)


def calcular_phi_estilo_guia(tiempo, canal_lvdt, canal_prox, frecuencia):
    """Implementa literalmente la ecuación de la guía:
    φ = -(Δt/T)×360° - 90°, con Δt = t_cruce0(LVDT) - t_flanco(Prox)
    """
    t_flancos, _ = detectar_flancos_rising(tiempo, canal_prox)
    t_cruces = detectar_cruces_cero_asc(tiempo, canal_lvdt)

    if len(t_flancos) < 2 or len(t_cruces) < 2:
        return None

    # Alinear eventos usando el segundo de cada uno para evitar transitorios y/o
    # emparejar por cercanía temporal
    pares = []
    for t_f in t_flancos:
        # buscar cruce de cero más cercano en el entorno de t_f
        idx = np.searchsorted(t_cruces, t_f)
        vecinos = []
        if idx > 0:
            vecinos.append(t_cruces[idx-1])
        if idx < len(t_cruces):
            vecinos.append(t_cruces[idx])
        if vecinos:
            t_c = min(vecinos, key=lambda v: abs(v - t_f))
            pares.append((t_c, t_f))

    if len(pares) == 0:
        return None

    T = 1.0 / frecuencia
    phis = []
    deltas = []
    for t_c, t_f in pares:
        delta_t = t_c - t_f
        # Estilo guía con +90° (equivalente a +π/2 rad)
        phi = -(delta_t / T) * 360.0 + 90.0
        # Normalizar a [-180, 180]
        while phi > 180.0:
            phi -= 360.0
        while phi < -180.0:
            phi += 360.0
        phis.append(phi)
        deltas.append(delta_t)

    phi_deg = float(np.median(phis))
    phi_rad = np.deg2rad(phi_deg)
    delta_t_med = float(np.median(deltas))

    return {
        'delta_t': delta_t_med,
        'T': T,
        'phi_rad': phi_rad,
        'phi_deg': phi_deg,
        'pares_usados': len(phis),
        'metodo': 'guia'
    }


def procesar_archivo(filepath, frecuencia):
    """
    Procesa un archivo CSV y calcula φ con ambos métodos
    """
    tiempo, canal_lvdt, canal_prox = leer_csv(filepath)
    
    if tiempo is None:
        return None
    
    # Método 1: Máximos (ambas señales)
    resultado_max = calcular_phi_maximos(tiempo, canal_lvdt, canal_prox, frecuencia)
    
    # Método 2: Correlación cruzada
    resultado_corr = calcular_phi_correlacion(tiempo, canal_lvdt, canal_prox, frecuencia)
    
    # Método 3: Pulso (prox) vs. máximo (LVDT)
    resultado_pulso = calcular_phi_pulso_vs_maximo(tiempo, canal_lvdt, canal_prox, frecuencia)
    
    # Método 4: Estilo guía (cruce LVDT vs flanco prox)
    resultado_guia = calcular_phi_estilo_guia(tiempo, canal_lvdt, canal_prox, frecuencia)
    
    return {
        'maximos': resultado_max,
        'correlacion': resultado_corr,
        'pulso_max': resultado_pulso,
        'guia': resultado_guia
    }


def main():
    """
    Procesa todos los archivos y genera tabla de resultados
    """
    print("="*80)
    print("CÁLCULO MEJORADO DEL ÁNGULO DE DESFASE φ")
    print("Método 1: Tiempo entre máximos")
    print("Método 2: Correlación cruzada")
    print("Método 3: Pulso (prox) vs Máximo (LVDT)")
    print("Método 4: Estilo guía (cruce LVDT - flanco prox)")
    print("="*80)
    print()
    
    resultados = []
    
    for carpeta, config in configuraciones.items():
        freq = config['freq']
        tipo = config['tipo']
        
        print(f"\n{'─'*80}")
        print(f"Procesando: {carpeta} (f = {freq} Hz, {tipo})")
        print(f"{'─'*80}")
        
        carpeta_path = base_path / carpeta
        
        if not carpeta_path.exists():
            print(f"  ⚠ Carpeta no encontrada: {carpeta_path}")
            continue
        
        archivos = sorted(carpeta_path.glob("*.csv"))
        
        if len(archivos) == 0:
            print(f"  ⚠ No se encontraron archivos CSV")
            continue
        
        phis_max = []
        phis_corr = []
        phis_pulso = []
        phis_guia = []
        
        for archivo in archivos:
            resultado = procesar_archivo(archivo, freq)
            
            if resultado is not None:
                print(f"\n  Archivo: {archivo.name}")
                
                if resultado['maximos'] is not None:
                    r_max = resultado['maximos']
                    print(f"    MÉTODO MÁXIMOS:")
                    print(f"      Δt = {r_max['delta_t']*1000:.3f} ms")
                    print(f"      φ  = {r_max['phi_deg']:.2f}°")
                    phis_max.append(r_max['phi_deg'])
                
                if resultado['correlacion'] is not None:
                    r_corr = resultado['correlacion']
                    print(f"    MÉTODO CORRELACIÓN:")
                    print(f"      Δt = {r_corr['delta_t']*1000:.3f} ms")
                    print(f"      φ  = {r_corr['phi_deg']:.2f}°")
                    phis_corr.append(r_corr['phi_deg'])
                
                if resultado.get('pulso_max') is not None:
                    r_pul = resultado['pulso_max']
                    print(f"    MÉTODO PULSO vs MÁXIMO:")
                    print(f"      Pares usados = {r_pul['pares_usados']}")
                    print(f"      Δt = {r_pul['delta_t']*1000:.3f} ms")
                    print(f"      φ  = {r_pul['phi_deg']:.2f}°")
                    phis_pulso.append(r_pul['phi_deg'])
                
                if resultado.get('guia') is not None:
                    r_g = resultado['guia']
                    print(f"    MÉTODO ESTILO GUÍA:")
                    print(f"      Pares usados = {r_g['pares_usados']}")
                    print(f"      Δt = {r_g['delta_t']*1000:.3f} ms")
                    print(f"      φ  = {r_g['phi_deg']:.2f}°")
                    phis_guia.append(r_g['phi_deg'])
                
                # Guardar para tabla
                resultados.append({
                    'Configuración': tipo,
                    'Frecuencia (Hz)': freq,
                    'Archivo': archivo.name,
                    'φ_max (°)': r_max['phi_deg'] if resultado['maximos'] else np.nan,
                    'φ_corr (°)': r_corr['phi_deg'] if resultado['correlacion'] else np.nan,
                    'φ_pulso (°)': r_pul['phi_deg'] if resultado.get('pulso_max') else np.nan,
                    'φ_guia (°)': r_g['phi_deg'] if resultado.get('guia') else np.nan,
                })
        
        # Promedios
        if len(phis_max) > 0:
            print(f"\n  ✓ MÁXIMOS:     φ promedio = {np.mean(phis_max):.2f}° ± {np.std(phis_max):.2f}°")
        if len(phis_corr) > 0:
            print(f"  ✓ CORRELACIÓN: φ promedio = {np.mean(phis_corr):.2f}° ± {np.std(phis_corr):.2f}°")
        if len(phis_pulso) > 0:
            print(f"  ✓ PULSO-MÁX:   φ promedio = {np.mean(phis_pulso):.2f}° ± {np.std(phis_pulso):.2f}°")
        if len(phis_guia) > 0:
            print(f"  ✓ ESTILO GUÍA: φ promedio = {np.mean(phis_guia):.2f}° ± {np.std(phis_guia):.2f}°")
    
    # Crear tabla resumen
    if len(resultados) > 0:
        df_resultados = pd.DataFrame(resultados)
        
        print("\n" + "="*80)
        print("RESUMEN POR CONFIGURACIÓN Y FRECUENCIA")
        print("="*80)
        
        resumen = df_resultados.groupby(['Configuración', 'Frecuencia (Hz)']).agg({
            'φ_max (°)': ['mean', 'std', 'count'],
            'φ_corr (°)': ['mean', 'std'],
            'φ_pulso (°)': ['mean', 'std'],
            'φ_guia (°)': ['mean', 'std']
        }).round(2)
        
        print(resumen)
        
        # Guardar
        output_file = base_path.parent / "resultados_phi_mejorado.csv"
        df_resultados.to_csv(output_file, index=False, sep=';', decimal=',')
        print(f"\n✓ Resultados guardados en: {output_file}")
    else:
        print("\n⚠ No se procesaron archivos exitosamente")


if __name__ == "__main__":
    main()
