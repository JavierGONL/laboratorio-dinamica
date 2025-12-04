"""
Generación de gráficas de señales temporales para experimentos de vibración

PROPÓSITO:
-----------
Este script procesa múltiples archivos CSV de un experimento de vibración y genera
gráficas continuas de las señales registradas por los sensores a lo largo del tiempo.
Puede procesar una carpeta completa con varios archivos CSV y concatenarlos en una
sola gráfica continua, ideal para visualizar la evolución temporal de oscilaciones.

FUNCIONALIDADES:
----------------
1. Lee múltiples archivos CSV de una carpeta en orden secuencial
2. Concatena las señales eliminando solapamientos y transitorios entre archivos
3. Genera gráficas suavizadas (filtro de media móvil) para mejor visualización
4. Produce tres tipos de gráficas:
   - Canal A solo: Señal del LVDT (voltaje)
   - Canal B solo: Señal del sensor de proximidad o adicional (milivoltaje)
   - Combinada A+B: Ambos canales con ejes Y independientes (opcional)

ARQUITECTURA DE CONCATENACIÓN:
-------------------------------
- Elimina solapamiento entre archivos (overlap_n puntos al inicio de cada segmento)
- Alinea el tiempo de cada segmento con el final del anterior
- Aplica delay ajustable entre segmentos para compensar discontinuidades
- Mantiene el orden cronológico de los archivos

FORMATO DE ENTRADA:
-------------------
- Archivos CSV con formato:
  * Separador: ';'
  * Decimal: ','
  * Opcional: skiprows=1 (saltar encabezado)
  * Columnas: [Tiempo(ms), Canal_A(V), Canal_B(mV)]
- Puede procesar archivos con 1, 2 o 3 columnas
- Maneja valores infinitos y NaN automáticamente

ARCHIVOS DE SALIDA:
-------------------
- todos_A.png: Gráfica del Canal A (LVDT)
- todos_B.png: Gráfica del Canal B (si existe)
- todos_A_B_combinados.png: Gráfica con ambos canales (si combine_channels=True)
- Resolución: 300 DPI
- Formato: PNG con fondo transparente

PROCESAMIENTO DE SEÑAL:
-----------------------
- Filtro de suavizado: Media móvil con ventana de 15 muestras
- Conversión de tiempo: ms → segundos para ejes de gráficas
- Normalización opcional para comparación de fases
- Eliminación de discontinuidades en bordes de archivos

PARÁMETROS CONFIGURABLES:
-------------------------
- path: Ruta a carpeta con CSVs o archivo individual
- delay: Retardo en ms entre segmentos (para ajustar concatenación)
- combine_channels: True para generar gráfica combinada A+B
- overlap_n: Número de muestras a eliminar en solapamiento (default: 400)

USO:
----
1. Modificar la variable 'path' con la ruta del experimento:
   path = r"G12\Vibracio libre"  # Ejemplo

2. Ajustar parámetros según necesidad:
   delay = 100  # ms entre segmentos
   combine_channels = False  # True para gráfica combinada

3. Ejecutar: python graficas_laboratorio.py

4. Las imágenes se guardan en la misma carpeta de los CSVs

APLICACIONES:
-------------
- Visualización de vibración libre (decaimiento exponencial)
- Análisis de vibración forzada en régimen permanente
- Identificación de transitorios y estabilización
- Documentación visual para informes de laboratorio

EJEMPLOS DE USO:
----------------
# Vibración libre con múltiples CSVs
path = r"G12\Vibracio libre"
delay = 0.0
combine_channels = False

# Vibración forzada con análisis de fase
path = r"G12\Forzada-amortiguada4.3hz"
delay = 50.0
combine_channels = True

NOTA:
-----
Para análisis cuantitativo de fase y amplitud, usar los scripts
especializados: calcular_phi.py y visualizar_senales.py
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def to_float_series(series):
    s = series.astype(str).str.strip()
    # Reemplazar decimal coma por punto
    s = s.str.replace(',', '.', regex=False)
    # Normalizar representaciones de infinito
    s = s.str.replace('+inf', 'inf', case=False, regex=False)
    s = s.str.replace('-inf', '-inf', case=False, regex=False)
    # Eliminar posibles separadores de miles si existieran
    s = s.str.replace('\u00A0', '', regex=False).str.replace(' ', '', regex=False)
    return pd.to_numeric(s, errors='coerce')


def osciloscope_plot(path, delay=0.0, combine_channels=False):
    try:
        print("Iniciando generación de gráfica...")

        if not os.path.exists(path):
            print(f"Error: La ruta {path} no existe")
            return

        # Determinar si es carpeta o archivo
        if os.path.isdir(path):
            csv_paths = sorted(
                [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')]
            )
            print(f"Archivos encontrados: {len(csv_paths)} archivos")
        else:
            csv_paths = [path]
            print("Se detectó un solo archivo CSV")

        if len(csv_paths) == 0:
            print("Error: No se encontraron archivos CSV")
            return

        # Ya no se acumulan todos los archivos, se grafica cada uno por separado
        b_present_in_all = True  # Será False si algún archivo no tiene la tercera columna
        tiempo_total = []
        canal_a_total = []
        canal_b_total = []

        # Usar todos los archivos CSV en orden
        if len(csv_paths) < 1:
            print("No hay archivos CSV para procesar.")
            return
        files_to_use = csv_paths
        current_offset_ms = 0.0
        last_t = None
        n_segments = len(files_to_use)
        prev_dt = None
        overlap_n = 400
          # número de muestras a solapar/eliminar
        for idx, file_path in enumerate(files_to_use):
            nombre = os.path.basename(file_path)
            print(f"Procesando: {nombre}")
            try:
                df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=1)
            except Exception:
                df = pd.read_csv(file_path, sep=';', decimal=',')
            if df.shape[1] < 1:
                print(f"Advertencia: {nombre} no tiene columnas. Se omite.")
                continue
            print(f"Columnas: {df.columns.tolist()}")
            print(f"Primeras filas:\n{df.head()}\n")
            tiempo_col = to_float_series(df.iloc[:, 0])
            canal_a_col = to_float_series(df.iloc[:, 1]) if df.shape[1] >= 2 else None
            canal_b_col = to_float_series(df.iloc[:, 2]) if df.shape[1] >= 3 else None
            # Filtrar filas válidas
            if df.shape[1] == 1:
                valid_mask = (~tiempo_col.isna()) & np.isfinite(tiempo_col.to_numpy(dtype=float))
                tiempo_valid = tiempo_col[valid_mask].to_numpy(dtype=float)
                # Alinear inicio del segmento y encadenar con offset acumulado
                if len(tiempo_valid) > 0:
                    t0 = tiempo_valid[0]
                    # Si no es el primer segmento, eliminar la parte solapada
                    if idx > 0 and prev_dt is not None:
                        tiempo_valid = tiempo_valid - t0 + last_t - overlap_n*prev_dt
                        # Eliminar todos los puntos cuyo tiempo sea <= last_t
                        mask = tiempo_valid > last_t
                        tiempo_valid = tiempo_valid[mask]
                        canal_a_valid = [np.nan]*len(tiempo_valid)
                        canal_b_valid = [np.nan]*len(tiempo_valid)
                    else:
                        tiempo_valid = tiempo_valid - t0
                        canal_a_valid = [np.nan]*len(tiempo_valid)
                        canal_b_valid = [np.nan]*len(tiempo_valid)
                    # Si no es el último segmento, eliminar el último punto
                    if idx < n_segments - 1 and len(tiempo_valid) > 1:
                        tiempo_valid = tiempo_valid[:-1]
                        canal_a_valid = canal_a_valid[:-1]
                        canal_b_valid = canal_b_valid[:-1]
                tiempo_total.extend(tiempo_valid)
                canal_a_total.extend(canal_a_valid)
                canal_b_total.extend(canal_b_valid)
                if len(tiempo_valid) > 1:
                    prev_dt = np.median(np.diff(tiempo_valid))
                if len(tiempo_valid) > 0:
                    last_t = tiempo_valid[-1]
                continue
            # Si hay al menos dos columnas
            valid_mask = (~tiempo_col.isna()) & (~canal_a_col.isna())
            valid_mask = valid_mask & np.isfinite(tiempo_col.to_numpy(dtype=float)) & np.isfinite(canal_a_col.to_numpy(dtype=float))
            if df.shape[1] >= 3:
                valid_mask = valid_mask & (~pd.isna(canal_b_col)) & np.isfinite(canal_b_col.to_numpy(dtype=float))
            tiempo_valid = tiempo_col[valid_mask].to_numpy(dtype=float)
            canal_a_valid = canal_a_col[valid_mask].to_numpy(dtype=float)
            if df.shape[1] >= 3:
                canal_b_valid = canal_b_col[valid_mask].to_numpy(dtype=float)
            else:
                canal_b_valid = np.array([np.nan]*len(tiempo_valid))
            # Alinear inicio del segmento y encadenar con offset acumulado
            if len(tiempo_valid) > 0:
                t0 = tiempo_valid[0]
                tiempo_valid = tiempo_valid - t0
            # Si no es el primer segmento, solapar el inicio restando prev_dt
            if len(tiempo_valid) > 0:
                t0 = tiempo_valid[0]
                if idx > 0 and prev_dt is not None:
                    tiempo_valid = tiempo_valid - t0 + last_t - overlap_n*prev_dt - delay
                    # Eliminar todos los puntos cuyo tiempo sea <= last_t
                    mask = tiempo_valid > last_t
                    tiempo_valid = tiempo_valid[mask]
                    canal_a_valid = canal_a_valid[mask]
                    canal_b_valid = canal_b_valid[mask]
                else:
                    tiempo_valid = tiempo_valid - t0
                # Si no es el último segmento, eliminar el último punto
                if idx < n_segments - 1 and len(tiempo_valid) > 1:
                    tiempo_valid = tiempo_valid[:-1]
                    canal_a_valid = canal_a_valid[:-1]
                    canal_b_valid = canal_b_valid[:-1]
            tiempo_total.extend(tiempo_valid)
            canal_a_total.extend(canal_a_valid)
            canal_b_total.extend(canal_b_valid)
            if len(tiempo_valid) > 1:
                prev_dt = np.median(np.diff(tiempo_valid))
            if len(tiempo_valid) > 0:
                last_t = tiempo_valid[-1]

        # Graficar todo junto
        if len(tiempo_total) == 0:
            print("Error: No se pudieron leer datos válidos")
            return
        tiempo_total = np.asarray(tiempo_total)
        canal_a_total = np.asarray(canal_a_total)
        canal_b_total = np.asarray(canal_b_total)
        # No reordenar: mantener la secuencia por archivo
        tiempo_s = (tiempo_total - tiempo_total[0]) / 1000.0
        def smooth(y, window=11):
            if window < 3 or window > len(y):
                return y
            kernel = np.ones(window) / window
            return np.convolve(y, kernel, mode='same')
        # Gráfica Canal A
        fig_a, ax_a = plt.subplots(figsize=(14, 7))
        ax_a.plot(tiempo_s, smooth(canal_a_total, window=15), linewidth=1.2, color='tab:blue', antialiased=True)
        ax_a.set_xlabel('Tiempo (s)', fontsize=12)
        ax_a.set_ylabel('Canal A (V)', fontsize=12)
        ax_a.set_title('Canal A vs Tiempo - TODOS LOS CSV', fontsize=14, fontweight='bold')
        ax_a.grid(True, alpha=0.25, linestyle='--')
        ax_a.margins(x=0)
        fig_a.tight_layout()
        output_path_a = os.path.join(os.path.dirname(files_to_use[0]), "todos_A.png")
        fig_a.savefig(output_path_a, dpi=300, bbox_inches='tight')
        plt.close(fig_a)
        print(f"\n✓ Gráfica Canal A generada (todos los CSV juntos)!")
        print(f"Archivo guardado en: {output_path_a}")
        
        # Gráfica Canal B (si existe)
        if not np.all(np.isnan(canal_b_total)):
            fig_b, ax_b = plt.subplots(figsize=(14, 7))
            ax_b.plot(tiempo_s, smooth(canal_b_total, window=15), linewidth=1.2, color='tab:red', antialiased=True)
            ax_b.set_xlabel('Tiempo (s)', fontsize=12)
            ax_b.set_ylabel(' señal (mV)', fontsize=12)
            ax_b.set_title('Vibracion Libre', fontsize=14, fontweight='bold')
            ax_b.grid(True, alpha=0.25, linestyle='--')
            ax_b.margins(x=0)
            fig_b.tight_layout()
            output_path_b = os.path.join(os.path.dirname(files_to_use[0]), "todos_B.png")
            fig_b.savefig(output_path_b, dpi=300, bbox_inches='tight')
            plt.close(fig_b)
            print(f"✓ Gráfica Canal B generada (todos los CSV)!")
            print(f"Archivo guardado en: {output_path_b}")
            
            # Gráfica combinada (ambos canales) si se solicita
            if combine_channels:
                fig_combined, ax_combined = plt.subplots(figsize=(14, 7))
                ax1 = ax_combined
                ax2 = ax_combined.twinx()
                
                line1 = ax1.plot(tiempo_s, smooth(canal_a_total, window=15), linewidth=1.2, color='tab:blue', label='Canal A', antialiased=True)
                line2 = ax2.plot(tiempo_s, smooth(canal_b_total, window=15), linewidth=1.2, color='tab:red', label='Canal B', antialiased=True)
                
                ax1.set_xlabel('Tiempo (s)', fontsize=12)
                ax1.set_ylabel('Canal A (V)', fontsize=12, color='tab:blue')
                ax2.set_ylabel('Canal B (mV)', fontsize=12, color='tab:red')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                ax1.set_title('Canales A y B vs Tiempo - TODOS LOS CSV', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.25, linestyle='--')
                ax1.margins(x=0)
                
                # Leyenda combinada
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right')
                
                fig_combined.tight_layout()
                output_path_combined = os.path.join(os.path.dirname(files_to_use[0]), "todos_A_B_combinados.png")
                fig_combined.savefig(output_path_combined, dpi=300, bbox_inches='tight')
                plt.close(fig_combined)
                print(f"✓ Gráfica combinada (A+B) generada!")
                print(f"Archivo guardado en: {output_path_combined}")
        else:
            print("Nota: Canal B incompleto o ausente en ambos archivos; se omitió su gráfica.")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    path = r"G12\Vibracio libre" # Ruta del csv
    delay = 100 # <-- Delay en ms entre segmentos (ajusta según necesites)
    combine_channels = False # <-- Cambiar a True para graficar ambos canales juntos
    osciloscope_plot(path, delay, combine_channels)