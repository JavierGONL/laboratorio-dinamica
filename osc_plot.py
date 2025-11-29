import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(',', '.', regex=False)
    s = s.str.replace('+inf', 'inf', case=False, regex=False)
    s = s.str.replace('-inf', '-inf', case=False, regex=False)
    s = s.str.replace('\u00A0', '', regex=False).str.replace(' ', '', regex=False)
    return pd.to_numeric(s, errors='coerce')


def read_csv_osc(file_path: str) -> pd.DataFrame:
    # Intentar con encabezado en primera fila (skiprows=1), formato europeo
    for skip in (1, 0):
        try:
            df = pd.read_csv(file_path, sep=';', decimal=',', skiprows=skip)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    raise ValueError(f"No se pudo leer correctamente el CSV: {file_path}")


def collect_series(paths: list, smooth_window: int = 0, inf_strategy: str = 'interp'):
    tiempo_total = []
    a_total = []
    b_total = []
    b_present_in_all = True
    offset_ms = 0.0

    for idx, p in enumerate(paths):
        df = read_csv_osc(p)
        # Seleccionar primeras 3 columnas si existen
        t_col = to_float_series(df.iloc[:, 0])
        a_col = to_float_series(df.iloc[:, 1])
        b_col = to_float_series(df.iloc[:, 2]) if df.shape[1] >= 3 else None
        if b_col is None:
            b_present_in_all = False

        # Validar tiempo siempre finito y no NaN
        t_valid_mask = (~t_col.isna()) & np.isfinite(t_col.to_numpy())
        t_val = t_col[t_valid_mask].to_numpy(dtype=float)

        # Manejo de inf/NaN en canales: 'drop' elimina filas; 'interp' interpola valores sobre tiempo válido
        def handle_channel(series: pd.Series, base_mask: np.ndarray) -> np.ndarray:
            s = series.to_numpy(dtype=float)
            mask = base_mask.copy()
            if inf_strategy == 'drop':
                valid_channel = (~np.isnan(s)) & np.isfinite(s)
                mask = mask & valid_channel
                return series[mask].to_numpy(dtype=float)
            # Interpolación lineal sobre tiempo válido
            s_clean = s.copy()
            bad = (np.isnan(s_clean)) | (~np.isfinite(s_clean)) | (~base_mask)
            if np.all(bad):
                return series[base_mask].to_numpy(dtype=float)
            x = t_col.to_numpy(dtype=float)
            good = (~bad)
            # Limitar a rango de puntos buenos para evitar extrapolaciones extremas
            s_interp = np.interp(x, x[good], s_clean[good])
            return s_interp[base_mask]

        a_val = handle_channel(a_col, t_valid_mask)
        b_val = handle_channel(b_col, t_valid_mask) if b_col is not None else None

        if len(t_val) == 0:
            continue

        # Concatenar aplicando offset para continuidad temporal
        tiempo_total.extend(t_val + offset_ms)
        a_total.extend(a_val)
        if b_val is not None:
            b_total.extend(b_val)

        # Estimar dt del archivo para offset siguiente
        if len(t_val) > 1:
            dt = t_val[1] - t_val[0]
            offset_ms = (t_val[-1] + offset_ms) + dt
        else:
            offset_ms = (t_val[-1] + offset_ms) + 1000.0

    if len(tiempo_total) == 0:
        raise ValueError("No se obtuvieron datos válidos.")

    t = np.asarray(tiempo_total)
    a = np.asarray(a_total)
    b = np.asarray(b_total) if (b_total and b_present_in_all) else None

    # Ordenar por tiempo y rebasar a t=0
    order = np.argsort(t)
    t = t[order]
    a = a[order]
    if b is not None and len(b) == len(order):
        b = b[order]
    t_s = (t - t[0]) / 1000.0

    # Suavizado opcional
    def smooth(y, window):
        if window and window >= 3 and window < len(y):
            k = np.ones(window) / window
            return np.convolve(y, k, mode='same')
        return y

    # Reconstrucción para eliminar picos y cortes visibles
    def reconstruct_signal(t_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        if y_arr is None:
            return None
        y = y_arr.copy()
        # Detectar discontinuidades en tiempo: dt negativo o salto grande
        dt = np.diff(t_arr)
        if len(dt) == 0:
            return y
        dt_med = np.median(np.abs(dt)) if np.any(dt != 0) else 0.0
        disc = np.where((dt < 0) | (dt > 10 * (dt_med if dt_med > 0 else 1e-9)))[0]
        # Donde haya discontinuidad, suavizar alrededor
        for idx in disc:
            i0 = max(0, idx - 5)
            i1 = min(len(y) - 1, idx + 6)
            seg = y[i0:i1]
            # Reemplazar por interpolación lineal en la ventana
            xseg = t_arr[i0:i1]
            good = np.isfinite(seg)
            if np.sum(good) >= 2:
                y[i0:i1] = np.interp(xseg, xseg[good], seg[good])
        # Detectar picos (derivadas anómalas) y corregir
        dy = np.diff(y)
        med = np.median(np.abs(dy)) if np.any(dy != 0) else 0.0
        thresh = 6 * (med if med > 0 else 1e-9)
        spikes = np.where(np.abs(dy) > thresh)[0]
        for idx in spikes:
            j0 = max(0, idx - 2)
            j1 = min(len(y) - 1, idx + 3)
            xseg = t_arr[j0:j1]
            yseg = y[j0:j1]
            good = np.isfinite(yseg)
            if np.sum(good) >= 2:
                y[j0:j1] = np.interp(xseg, xseg[good], yseg[good])
        return y

    a_rec = reconstruct_signal(t_s, a)
    b_rec = reconstruct_signal(t_s, b)
    a_plot = smooth(a_rec, smooth_window)
    b_plot = smooth(b_rec, smooth_window) if b_rec is not None else None

    return t_s, a_plot, b_plot


def estimate_damped_sine(t_s: np.ndarray, y: np.ndarray):
    if y is None or len(y) == 0:
        return 1.0, 0.0, 1.0, 0.0
    yc = y - np.nanmean(y)
    # Estimar frecuencia por FFT (más robusto)
    n = len(yc)
    dt = np.median(np.diff(t_s)) if len(t_s) > 1 else 0.001
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(yc))
    if len(freqs) > 1:
        peak_idx = np.argmax(spectrum[1:]) + 1
        f = freqs[peak_idx]
    else:
        f = 1.0
    # Búsqueda de delta por rejilla y ajuste lineal de fase
    deltas = np.linspace(0.0, 5.0, 51)  # decaimiento por segundo
    best_err = np.inf
    best_params = (1.0, 0.0)
    omega = 2.0 * np.pi * f
    for delta in deltas:
        # y * e^{delta t} ≈ C * sin(omega t) + D * cos(omega t)
        w = np.exp(delta * t_s)
        Y = yc * w
        S = np.sin(omega * t_s)
        Cc = np.cos(omega * t_s)
        X = np.column_stack([S, Cc])
        # Resolver mínimos cuadrados para C,D
        try:
            coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except Exception:
            continue
        Y_hat = X @ coeffs
        err = np.nanmean((Y - Y_hat) ** 2)
        if err < best_err:
            best_err = err
            best_params = (delta, coeffs)
    delta_opt, (C_opt, D_opt) = best_params
    A0 = float(np.hypot(C_opt, D_opt))
    phi = float(np.arctan2(D_opt, C_opt))
    return A0, delta_opt, f, phi


def simulate_signal(t_s: np.ndarray, y: np.ndarray):
    A0, delta, f, phi = estimate_damped_sine(t_s, y)
    sim = A0 * np.exp(-delta * t_s) * np.sin(2.0 * np.pi * f * t_s + phi)
    return sim


def simulate_signal_two_modes(t_s: np.ndarray, y: np.ndarray):
    if y is None or len(y) == 0:
        return None
    
    yc = y - np.mean(y)
    n = len(yc)
    dt = np.median(np.diff(t_s)) if len(t_s) > 1 else 0.001
    
    # 1. Estimar frecuencia dominante con FFT (método más confiable)
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.abs(np.fft.rfft(yc))
    # Excluir DC y encontrar pico principal
    peak_idx = np.argmax(spectrum[1:]) + 1
    f_est = freqs[peak_idx]
    omega = 2.0 * np.pi * f_est
    
    # 2. Calcular envolvente superior e inferior
    # Envolvente superior
    env_high = np.zeros_like(yc)
    env_low = np.zeros_like(yc)
    window_size = max(3, int(0.5 / (f_est * dt))) if f_est > 0 else 10
    
    for i in range(n):
        i_start = max(0, i - window_size)
        i_end = min(n, i + window_size + 1)
        env_high[i] = np.max(yc[i_start:i_end])
        env_low[i] = np.min(yc[i_start:i_end])
    
    # Amplitud instantánea
    A_inst = (env_high - env_low) / 2.0
    
    # 3. Ajustar decaimiento exponencial a la amplitud instantánea
    # Usar solo puntos con amplitud significativa
    threshold = 0.1 * np.max(A_inst)
    valid_idx = A_inst > threshold
    
    if np.sum(valid_idx) > 10:
        t_valid = t_s[valid_idx]
        A_valid = A_inst[valid_idx]
        log_A = np.log(A_valid + 1e-9)
        
        # Ajuste lineal ponderado (más peso a datos iniciales)
        weights = np.exp(-0.2 * t_valid)
        coeffs = np.polyfit(t_valid, log_A, 1, w=weights)
        delta = max(-coeffs[0], 0.0)
        A0 = np.exp(coeffs[1])
    else:
        A0 = np.max(A_inst)
        delta = 0.5
    
    # 4. Encontrar fase óptima por correlación cruzada
    # Generar candidatos de fase
    phases = np.linspace(-np.pi, np.pi, 91)
    correlations = []
    
    # Usar primera mitad de datos para optimización
    n_opt = min(n, int(0.5 * n))
    t_opt = t_s[:n_opt]
    y_opt = yc[:n_opt]
    
    for phi in phases:
        candidate = A0 * np.exp(-delta * t_opt) * np.sin(omega * t_opt + phi)
        # Correlación normalizada
        corr = np.sum(y_opt * candidate) / (np.sqrt(np.sum(y_opt**2) * np.sum(candidate**2)) + 1e-9)
        correlations.append(corr)
    
    best_phi = phases[np.argmax(correlations)]
    
    # 5. Generar señal simulada final
    sim = A0 * np.exp(-delta * t_s) * np.sin(omega * t_s + best_phi)
    
    return sim


def plot_series(t_s, a_plot, b_plot, out_dir: str, simulate: bool = False, compare: bool = False, multi: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    # Canal A
    fig_a, ax_a = plt.subplots(figsize=(14, 7))
    if simulate:
        a_sim = simulate_signal_two_modes(t_s, a_plot) if multi else simulate_signal(t_s, a_plot)
        ax_a.plot(t_s, a_sim, linewidth=1.6, color='tab:blue', antialiased=True, label='Simulación')
        if compare:
            ax_a.plot(t_s, a_plot, linewidth=1.0, color='tab:orange', alpha=0.6, label='Datos')
        ax_a.legend()
    else:
        ax_a.plot(t_s, a_plot, linewidth=1.2, color='tab:blue', antialiased=True)
    ax_a.set_xlabel('Tiempo (s)', fontsize=12)
    ax_a.set_ylabel('Canal A (V)', fontsize=12)
    ax_a.set_title('Canal A vs Tiempo - Datos de Osciloscopio', fontsize=14, fontweight='bold')
    ax_a.grid(True, alpha=0.25, linestyle='--')
    ax_a.margins(x=0)
    fig_a.tight_layout()
    out_a = os.path.join(out_dir, 'angulo_vs_tiempo_A.png')
    fig_a.savefig(out_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)

    print(f"✓ Canal A guardado: {out_a}")

    # Canal B
    if b_plot is not None:
        fig_b, ax_b = plt.subplots(figsize=(14, 7))
        if simulate:
            b_sim = simulate_signal_two_modes(t_s, b_plot) if multi else simulate_signal(t_s, b_plot)
            ax_b.plot(t_s, b_sim, linewidth=1.6, color='tab:red', antialiased=True, label='Simulación')
            if compare:
                ax_b.plot(t_s, b_plot, linewidth=1.0, color='tab:pink', alpha=0.6, label='Datos')
            ax_b.legend()
        else:
            ax_b.plot(t_s, b_plot, linewidth=1.2, color='tab:red', antialiased=True)
        ax_b.set_xlabel('Tiempo (s)', fontsize=12)
        ax_b.set_ylabel('Canal B (mV)', fontsize=12)
        ax_b.set_title('Canal B vs Tiempo - Datos de Osciloscopio', fontsize=14, fontweight='bold')
        ax_b.grid(True, alpha=0.25, linestyle='--')
        ax_b.margins(x=0)
        fig_b.tight_layout()
        out_b = os.path.join(out_dir, 'angulo_vs_tiempo_B.png')
        fig_b.savefig(out_b, dpi=300, bbox_inches='tight')
        plt.close(fig_b)
        print(f"✓ Canal B guardado: {out_b}")
    else:
        print("Nota: Canal B incompleto o ausente; no se graficó.")


def list_csvs(path: str) -> list:
    if os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')])
        return files
    return [path]


def main():
    parser = argparse.ArgumentParser(description='Graficar datos de osciloscopio (CSV sep=; decimal=,)')
    parser.add_argument('--path', required=True, help='Ruta a carpeta con CSV o a un CSV')
    parser.add_argument('--out', default=os.getcwd(), help='Carpeta de salida de las imágenes')
    parser.add_argument('--smooth', type=int, default=7, help='Ventana de suavizado (media móvil). 0 para desactivar')
    parser.add_argument('--inf-strategy', choices=['drop','interp'], default='interp', help='Cómo tratar NaN/±inf en señales: drop (eliminar) o interp (interpolar)')
    parser.add_argument('--simulate', action='store_true', default=True, help='Graficar una simulación (seno amortiguado) ajustada a los datos')
    parser.add_argument('--compare', action='store_true', default=True, help='Mostrar simulación y datos experimentales juntos con leyenda')
    parser.add_argument('--simulate-multi', action='store_true', default=True, help='Usar dos modos amortiguados (dos frecuencias dominantes) en la simulación')
    args = parser.parse_args()

    path = args.path
    out = args.out
    smooth_w = max(0, args.smooth)

    if not os.path.exists(path):
        print(f"Error: La ruta no existe: {path}")
        sys.exit(1)

    csvs = list_csvs(path)
    if len(csvs) == 0:
        print("Error: No se encontraron CSVs en la ruta proporcionada")
        sys.exit(1)

    print(f"Procesando {len(csvs)} archivo(s) CSV...")
    try:
        t_s, a_plot, b_plot = collect_series(csvs, smooth_window=smooth_w, inf_strategy=args.inf_strategy)
        plot_series(t_s, a_plot, b_plot, out, simulate=args.simulate, compare=args.compare, multi=args.simulate_multi)
        print("\n✓ Gráficas generadas correctamente")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # MODIFICADO: El usuario debe colocar aquí el path manualmente
    # Ejemplo:
    # path = r'C:/Users/lunit/OneDrive/Desktop/laboratorio dinamica/G12/Forzada-amortiguada4.2hz/'
    path = 'C:\Users\lunit\OneDrive\Desktop\laboratorio dinamica\G12\Libre-Amortiguiado-Abierta.psdata' 
    out = os.getcwd()
    smooth_w = 0  # Cambia si quieres otra ventana de suavizado
    inf_strategy = 'interp'  # 'drop' o 'interp'
    simulate = False
    compare = True
    multi = True  # True para dos modos, False para uno

    if not path:
        print('Por favor, edita osc_plot.py y coloca la ruta en la variable path.')
        sys.exit(1)

    if not os.path.exists(path):
        print(f"Error: La ruta no existe: {path}")
        sys.exit(1)

    csvs = list_csvs(path)
    if len(csvs) == 0:
        print("Error: No se encontraron CSVs en la ruta proporcionada")
        sys.exit(1)

    print(f"Procesando {len(csvs)} archivo(s) CSV...")
    try:
        t_s, a_plot, b_plot = collect_series(csvs, smooth_window=smooth_w, inf_strategy=inf_strategy)
        plot_series(t_s, a_plot, b_plot, out, simulate=simulate, compare=compare, multi=multi)
        print("\n✓ Gráficas generadas correctamente")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
