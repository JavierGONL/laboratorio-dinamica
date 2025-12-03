import pandas as pd
import numpy as np
from pathlib import Path

# Leer un archivo de cada configuración
cerrada = pd.read_csv(r'G12\Forzada-amortiguada4.2hz\Forzada-amortiguada4.2hz_1.csv', sep=';', skiprows=2, decimal=',')
abierta = pd.read_csv(r'G12\Forzada-amortiguada-abierta4.2hz\Forzada-amortiguada-abierta4.2hz_1.csv', sep=';', skiprows=2, decimal=',')

# Limpieza: valores -∞ → NaN
for df in [cerrada, abierta]:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# Canal A (voltaje) es el desplazamiento (nombre puede tener espacios)
print("Columnas cerrada:", cerrada.columns.tolist())
print("Columnas abierta:", abierta.columns.tolist())
canal_a_cerrada = cerrada.iloc[:, 1].dropna()  # Segunda columna (índice 1)
canal_a_abierta = abierta.iloc[:, 1].dropna()

# Calcular amplitudes (pico a pico / 2)
amp_cerrada = (canal_a_cerrada.max() - canal_a_cerrada.min()) / 2.0
amp_abierta = (canal_a_abierta.max() - canal_a_abierta.min()) / 2.0

print("=== Análisis de amplitudes en forzada amortiguada a 4.2 Hz ===")
print(f"Configuración 'cerrada' (Forzada-amortiguada4.2hz):")
print(f"  Amplitud pico-pico/2: {amp_cerrada:.6f} V")
print(f"\nConfiguración 'abierta' (Forzada-amortiguada-abierta4.2hz):")
print(f"  Amplitud pico-pico/2: {amp_abierta:.6f} V")
print(f"\nRelación amplitudes abierta/cerrada: {amp_abierta/amp_cerrada:.3f}")

# La configuración con MENOR amplitud tiene MÁS amortiguamiento (mayor ζ)
# La configuración con MAYOR amplitud tiene MENOS amortiguamiento (menor ζ)
if amp_abierta > amp_cerrada:
    print("\n*** CONCLUSIÓN: 'abierta' tiene MENOR amortiguamiento (ζ pequeño) → FAmax alto ***")
    print("    'cerrada' tiene MAYOR amortiguamiento (ζ grande) → FAmax bajo")
else:
    print("\n*** CONCLUSIÓN: 'abierta' tiene MAYOR amortiguamiento (ζ grande) → FAmax bajo ***")
    print("    'cerrada' tiene MENOR amortiguamiento (ζ pequeño) → FAmax alto")
    print("\n¡LAS ETIQUETAS ESTÁN INVERTIDAS!")
