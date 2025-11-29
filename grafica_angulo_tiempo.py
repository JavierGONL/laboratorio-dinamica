import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import numpy as np
import os

# Ruta a la carpeta con los archivos CSV
folder_path = r"c:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\G12\Vibracio libre"

# Leer todos los archivos CSV
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
files.sort()

# Crear la figura
plt.figure(figsize=(12, 8))

# Colores para cada archivo
colors = ['blue', 'red', 'green', 'orange', 'purple']

for idx, file in enumerate(files):
    file_path = os.path.join(folder_path, file)
    
    # Leer el CSV (saltar las primeras 2 filas que son encabezados)
    df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=1)
    
    # Renombrar columnas
    df.columns = ['Tiempo', 'Canal_A', 'Canal_B']
    
    # Convertir tiempo de ms a segundos
    tiempo_s = df['Tiempo'] / 1000
    
    # Usar Canal B como ángulo (en mV, puede necesitar conversión a grados)
    angulo = df['Canal_B']
    
    # Graficar
    plt.plot(tiempo_s, angulo, label=f'Medición {idx+1}', 
             color=colors[idx % len(colors)], alpha=0.7, linewidth=1)

plt.xlabel('Tiempo (s)', fontsize=12)
plt.ylabel('Ángulo (mV)', fontsize=12)
plt.title('Ángulo vs Tiempo - Vibración Libre', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar la figura
output_path = r"c:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\angulo_vs_tiempo.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Gráfica generada exitosamente: {output_path}")
plt.close()
