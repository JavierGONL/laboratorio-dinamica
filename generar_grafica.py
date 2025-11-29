import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    print("Iniciando generación de gráfica...")
    
    # Ruta a la carpeta con los archivos CSV
    folder_path = r"c:/Users/lunit\OneDrive\Desktop\laboratorio dinamica\G12\Vibracio libre"
    
    # Verificar que la carpeta existe
    if not os.path.exists(folder_path):
        print(f"Error: La carpeta {folder_path} no existe")
        exit(1)
    
    # Leer todos los archivos CSV
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    print(f"Archivos encontrados: {len(files)} archivos")
    
    if len(files) == 0:
        print("Error: No se encontraron archivos CSV")
        exit(1)
    
    # Listas para combinar todos los datos
    tiempo_total = []
    angulo_total = []
    
    offset_tiempo = 0  # Offset acumulado en milisegundos
    
    # Leer cada archivo y combinar los datos
    for idx, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        print(f"Procesando: {file}")
        
        # Leer el CSV con separador punto y coma y decimal coma
        df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=1)
        
        if idx == 0:
            print(f"Columnas: {df.columns.tolist()}")
            print(f"Primeras filas:\n{df.head()}")
        
        # Columnas: Tiempo (ms), Canal A (V), Canal B (mV)
        tiempo = df.iloc[:, 0].values + offset_tiempo  # Añadir offset acumulado
        angulo = df.iloc[:, 1].values  # Canal A como ángulo
        
        tiempo_total.extend(tiempo)
        angulo_total.extend(angulo)
        
        # Actualizar offset para el siguiente archivo (cada archivo es ~1 segundo)
        if len(tiempo) > 0:
            offset_tiempo = tiempo[-1] + (tiempo[1] - tiempo[0]) if len(tiempo) > 1 else offset_tiempo + 1000
    
    # Convertir a arrays numpy
    tiempo_total = np.array(tiempo_total)
    angulo_total = np.array(angulo_total)
    
    print(f"\nTotal de puntos: {len(tiempo_total)}")
    print(f"Rango de tiempo: {tiempo_total.min():.2f} ms a {tiempo_total.max():.2f} ms")
    
    # Crear la figura
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Graficar todos los datos continuos
    ax.plot(tiempo_total / 1000, angulo_total, linewidth=0.8, color='blue')  # Convertir a segundos
    
    ax.set_xlabel('Tiempo (s)', fontsize=12)
    ax.set_ylabel('Ángulo (V)', fontsize=12)
    ax.set_title('Ángulo vs Tiempo - Vibración Libre (Datos Continuos)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Guardar la figura
    output_path = r"c:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\angulo_vs_tiempo.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfica generada exitosamente!")
    print(f"Archivo guardado en: {output_path}")
    
    plt.close(fig)
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
