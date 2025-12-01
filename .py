import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Definir to_float_series al inicio para que esté disponible en todo el script
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


def osciloscope_plot(path):
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
        overlap_n = 300
        
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
                    tiempo_valid = tiempo_valid - t0 + last_t - overlap_n*prev_dt
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
        # Convertir Canal A de V a mV
        canal_a_total_mV = canal_a_total * 1000
        # Gráfica Canal A y B sobrepuestas (ambos en mV)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(tiempo_s, smooth(canal_a_total_mV, window=15), linewidth=1.2, color='tab:blue', label='Canal A (mV)', antialiased=True)
        if not np.all(np.isnan(canal_b_total)):
            ax.plot(tiempo_s, smooth(canal_b_total, window=15), linewidth=1.2, color='tab:red', label='Canal B (mV)', antialiased=True)
        ax.set_xlabel('Tiempo (s)', fontsize=12)
        ax.set_ylabel('Voltaje (mV)', fontsize=12)
        ax.set_title('Canales A y B (mV) vs Tiempo - TODOS LOS CSV', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.margins(x=0)
        ax.legend(fontsize=12)
        fig.tight_layout()
        output_path = os.path.join(os.path.dirname(files_to_use[0]), "todos_AyB.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n✓ Gráfica Canal A y B generada (todos los CSV juntos)!")
        print(f"Archivo guardado en: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ruta por defecto: carpeta con múltiples CSV de Libre-Amortiguiado-Abierta
    path = r"C:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\G12\Libre Amortiguada"
    osciloscope_plot(path)