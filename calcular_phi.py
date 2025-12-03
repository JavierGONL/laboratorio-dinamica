"""
Cálculo del ángulo de desfase φ:
φ = -(Δt/T)×2π + π/2   (convención guía: +90°)

Donde:
- Δt: Diferencia temporal entre flanco del LVDT y cruce por cero del sensor de proximidad
- T: Período de oscilación (T = 1/f)
- φ: Ángulo de desfase en radianes
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Configuración
base_path = Path(r"c:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\G12")

# Frecuencias y configuraciones a procesar
configuraciones = {
    'Forzada-amortiguada4.2hz': {'freq': 4.2, 'tipo': 'cerrada'},
    'Forzada-amortiguada4.3hz': {'freq': 4.3, 'tipo': 'cerrada'},
    'Forzada-amortiguada4.45hz': {'freq': 4.45, 'tipo': 'cerrada'},
    'Forzada-amortiguada-abierta4.2hz': {'freq': 4.2, 'tipo': 'abierta'},
    'Forzada-amortiguada-abierta4.3hz': {'freq': 4.3, 'tipo': 'abierta'},
    'Forzada-amortiguada-abierta4.45hz': {'freq': 4.45, 'tipo': 'abierta'},
}


def leer_csv(filepath):
    """
    Lee archivo CSV con formato:
    - Separador: ;
    - Decimal: ,
    - skiprows=2 (saltar encabezados)
    """
    try:
        df = pd.read_csv(filepath, sep=';', decimal=',', skiprows=2, header=None)
        # Columnas: 0=Tiempo(ms), 1=Canal A(V), 2=Canal B(mV)
        tiempo_ms = df.iloc[:, 0].values  # Tiempo en ms
        canal_a = df.iloc[:, 1].values    # LVDT - Señal CUADRADA (va hacia arriba)
        canal_b = df.iloc[:, 2].values    # Proximidad - Señal SINUSOIDAL
        
        # Convertir tiempo a segundos
        tiempo_s = tiempo_ms / 1000.0
        
        return tiempo_s, canal_a, canal_b
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return None, None, None


def detectar_flanco_lvdt(tiempo, canal_a):
    """
    Detecta el primer flanco ascendente de la señal del LVDT.
    Canal A es una señal CUADRADA que va hacia arriba.
    """
    # Normalizar la señal (puede tener offset)
    umbral = (np.max(canal_a) + np.min(canal_a)) / 2
    
    # Buscar flancos ascendentes (transición de bajo a alto)
    flancos = []
    for i in range(len(canal_a) - 1):
        if canal_a[i] < umbral and canal_a[i+1] >= umbral:
            # Interpolación para tiempo exacto
            t_flanco = tiempo[i] + (umbral - canal_a[i]) * (tiempo[i+1] - tiempo[i]) / (canal_a[i+1] - canal_a[i])
            flancos.append(t_flanco)
    
    if len(flancos) > 0:
        # Retornar el primer flanco después de estabilización
        if len(flancos) > 1:
            return flancos[1]  # Segundo flanco para evitar transitorios
        return flancos[0]
    return None


def detectar_cruce_cero_proximidad(tiempo, canal_b):
    """
    Detecta el primer cruce por cero ascendente de la señal del sensor de proximidad.
    Canal B es una señal SINUSOIDAL.
    Usa interpolación lineal para obtener el tiempo exacto.
    """
    # Buscar cruces por cero (de negativo a positivo)
    cruces = []
    for i in range(len(canal_b) - 1):
        if canal_b[i] <= 0 and canal_b[i+1] > 0:
            # Interpolación lineal para encontrar el tiempo exacto
            t_cruce = tiempo[i] - canal_b[i] * (tiempo[i+1] - tiempo[i]) / (canal_b[i+1] - canal_b[i])
            cruces.append(t_cruce)
    
    if len(cruces) > 0:
        # Retornar el primer cruce después de que la señal se estabilice
        if len(cruces) > 1:
            return cruces[1]  # Segundo cruce para evitar transitorios
        return cruces[0]
    return None


def calcular_phi(delta_t, T):
    """
    Calcula el ángulo de desfase según la guía:
    φ = -(Δt/T)×2π + π/2  (equivale a sumar +90°)

    Retorna φ en radianes y grados.
    """
    phi_rad = -(delta_t / T) * 2 * np.pi - (np.pi / 2)
    phi_deg = phi_rad * 180 / np.pi

    return phi_rad, phi_deg


def procesar_archivo(filepath, frecuencia):
    """
    Procesa un archivo CSV y calcula φ
    """
    tiempo, canal_a, canal_b = leer_csv(filepath)
    
    if tiempo is None:
        return None
    
    # Detectar eventos
    # Canal A = LVDT (señal cuadrada con flanco ascendente)
    # Canal B = Proximidad (señal sinusoidal con cruce por cero)
    t_flanco_lvdt = detectar_flanco_lvdt(tiempo, canal_a)
    t_cruce_prox = detectar_cruce_cero_proximidad(tiempo, canal_b)
    
    if t_flanco_lvdt is None or t_cruce_prox is None:
        print(f"  ⚠ No se pudieron detectar eventos en {os.path.basename(filepath)}")
        return None
    
    # Calcular Δt = tiempo_LVDT - tiempo_proximidad
    delta_t = t_flanco_lvdt - t_cruce_prox
    
    # Período
    T = 1 / frecuencia
    
    # Calcular φ
    phi_rad, phi_deg = calcular_phi(delta_t, T)
    
    return {
        't_flanco_lvdt': t_flanco_lvdt,
        't_cruce_prox': t_cruce_prox,
        'delta_t': delta_t,
        'T': T,
        'phi_rad': phi_rad,
        'phi_deg': phi_deg
    }


def main():
    """
    Procesa todos los archivos y genera tabla de resultados
    """
    print("="*80)
    print("CÁLCULO DEL ÁNGULO DE DESFASE φ")
    print("Fórmula: φ = -(Δt/T)×2π + π/2  (+=90°)")
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
        
        # Procesar todos los archivos CSV en la carpeta
        archivos = sorted(carpeta_path.glob("*.csv"))
        
        if len(archivos) == 0:
            print(f"  ⚠ No se encontraron archivos CSV en {carpeta}")
            continue
        
        phis_deg = []
        
        for archivo in archivos:
            resultado = procesar_archivo(archivo, freq)
            
            if resultado is not None:
                print(f"\n  Archivo: {archivo.name}")
                print(f"    t_flanco_LVDT   = {resultado['t_flanco_lvdt']*1000:.3f} ms")
                print(f"    t_cruce_prox    = {resultado['t_cruce_prox']*1000:.3f} ms")
                print(f"    Δt              = {resultado['delta_t']*1000:.3f} ms")
                print(f"    T               = {resultado['T']*1000:.3f} ms")
                print(f"    φ               = {resultado['phi_rad']:.4f} rad")
                print(f"    φ               = {resultado['phi_deg']:.2f}°")
                
                phis_deg.append(resultado['phi_deg'])
                
                resultados.append({
                    'Configuración': tipo,
                    'Frecuencia (Hz)': freq,
                    'Archivo': archivo.name,
                    'Δt (ms)': resultado['delta_t'] * 1000,
                    'φ (rad)': resultado['phi_rad'],
                    'φ (°)': resultado['phi_deg']
                })
        
        # Promedio para esta configuración/frecuencia
        if len(phis_deg) > 0:
            phi_promedio = np.mean(phis_deg)
            phi_std = np.std(phis_deg)
            print(f"\n  ✓ Promedio φ = {phi_promedio:.2f}° ± {phi_std:.2f}°")
    
    # Crear tabla resumen
    if len(resultados) > 0:
        df_resultados = pd.DataFrame(resultados)
        
        print("\n" + "="*80)
        print("TABLA RESUMEN DE RESULTADOS")
        print("="*80)
        print(df_resultados.to_string(index=False))
        
        # Guardar a CSV
        output_file = base_path.parent / "resultados_phi.csv"
        df_resultados.to_csv(output_file, index=False, sep=';', decimal=',')
        print(f"\n✓ Resultados guardados en: {output_file}")
        
        # Tabla resumen por configuración/frecuencia
        print("\n" + "="*80)
        print("TABLA RESUMEN POR CONFIGURACIÓN Y FRECUENCIA")
        print("="*80)
        
        resumen = df_resultados.groupby(['Configuración', 'Frecuencia (Hz)']).agg({
            'φ (°)': ['mean', 'std', 'count']
        }).round(2)
        
        print(resumen)
    else:
        print("\n⚠ No se procesaron archivos exitosamente")


if __name__ == "__main__":
    main()
