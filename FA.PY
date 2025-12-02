import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def main():
    # Datos experimentales (de la tabla del informe)
    r_exp_abierta = [0.6642, 0.6800, 0.7038]
    FA_exp_abierta = [1.7895, 1.8603, 1.9813]
    r_exp_cerrada = [0.6642, 0.6800, 0.7038]
    FA_exp_cerrada = [1.7896, 1.8603, 1.9814]

    # Parámetros (ζ = C/C_c asumidos adimensionales)
    # Zetas recalculadas por decremento log multi-n (promedio n=1..3)
    zeta_abierto = 0.00275
    zeta_cerrado = 0.026

    # Malla amplia para mostrar forma general (pico resonancia) hasta r=2
    r = np.linspace(0.3, 2.0, 1600)

    def FA_teorico(r, zeta):
        return 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * zeta * r)**2)
    def phi_teorico_deg(r, zeta):
        phi = np.degrees(np.arctan2(2.0 * zeta * r, 1.0 - r**2))
        return np.where(phi < 0.0, phi + 180.0, phi)

    FA_abierto = FA_teorico(r, zeta_abierto)
    FA_cerrado = FA_teorico(r, zeta_cerrado)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Curvas teóricas diferenciadas
    ax.plot(r, FA_abierto, label=f'Teórica abierta (ζ={zeta_abierto:.5f})', color='tab:blue', linewidth=2.0)
    ax.plot(r, FA_cerrado, label=f'Teórica cerrada (ζ={zeta_cerrado:.5f})', color='tab:orange', linewidth=2.0, linestyle='--')

    # Región donde están los puntos experimentales (pinto cerrada primero para no tapar abierta)
    ax.scatter(r_exp_cerrada, FA_exp_cerrada, color='purple', marker='s', edgecolor='black', linewidths=0.8,
               zorder=5, s=56, alpha=0.95, label='Exp. cerrada')
    ax.scatter(r_exp_abierta, FA_exp_abierta, facecolors='none', edgecolors='red', marker='o', linewidths=1.2,
               zorder=6, s=70, label='Exp. abierta')

    # Línea vertical resonancia
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.text(1.0 + 0.01, max(FA_abierto.max(), FA_cerrado.max()) * 0.85, 'resonancia',
            rotation=90, va='top', ha='left', color='gray', fontsize=9)

    # Escala principal: permitir ver el pico (que con ζ muy pequeño se hace enorme cerca de r=1)
    # Limitamos FA máximo para lectura (clipping visual) usando percentil para evitar distorsión extrema
    fa_limite = np.percentile(np.concatenate([FA_abierto, FA_cerrado]), 99.5)
    ax.set_ylim(0, fa_limite)
    # Limites simétricos alrededor de r=1 para centrar la línea de resonancia
    ax.set_xlim(0.6, 1.4)

    ax.set_xlabel(r'$\omega_f / \omega_n$')
    ax.set_ylabel(r'$FA$')
    ax.set_title(r'Factor de Amplificación')
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(loc='upper right')

    # Inset (zoom) para la zona experimental
    ax_zoom = inset_axes(ax, width="42%", height="52%", loc='lower left',
                         bbox_to_anchor=(0.05, 0.20, 0.5, 0.5), bbox_transform=ax.transAxes, borderpad=1)
    # Curvas en zoom (solo tramo local r 0.63-0.72)
    r_zoom = np.linspace(0.63, 0.72, 400)
    ax_zoom.plot(r_zoom, FA_teorico(r_zoom, zeta_abierto), color='tab:blue', linewidth=1.6)
    ax_zoom.plot(r_zoom, FA_teorico(r_zoom, zeta_cerrado), color='tab:orange', linewidth=1.6, linestyle='--')
    ax_zoom.scatter(r_exp_cerrada, FA_exp_cerrada, color='purple', marker='s', edgecolor='black', linewidths=0.7,
                    zorder=6, s=44, alpha=0.95)
    ax_zoom.scatter(r_exp_abierta, FA_exp_abierta, facecolors='none', edgecolors='red', marker='o', linewidths=1.0,
                    zorder=7, s=56)
    ax_zoom.set_xlim(0.63, 0.72)
    ax_zoom.set_ylim(min(FA_exp_abierta + FA_exp_cerrada) * 0.95,
                     max(FA_exp_abierta + FA_exp_cerrada) * 1.03)
    ax_zoom.set_xticks([0.64, 0.66, 0.68, 0.70, 0.72])
    ax_zoom.tick_params(labelsize=8)
    ax_zoom.grid(True, linestyle=':', alpha=0.3)
    ax_zoom.set_title('Zona experimental', fontsize=9)

    # Ajuste manual de márgenes evitando tight_layout (incompatible con inset)
    fig.subplots_adjust(top=0.92, bottom=0.1, left=0.09, right=0.98)
    output_path = 'FA_vs_r_abierta_cerrada.png'
    fig.savefig(output_path, dpi=300)
    print(f'Figura guardada en: {output_path}')
    try:
        img = plt.imread(output_path)
        print('Lectura verificada. Dimensiones:', img.shape)
    except Exception as e:
        print('Error al verificar la imagen:', e)

    # ---------------------- Figura de fase ----------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    phi_abierto = phi_teorico_deg(r, zeta_abierto)
    phi_cerrado = phi_teorico_deg(r, zeta_cerrado)
    ax2.plot(r, phi_abierto, label=f'Teórica abierta (ζ={zeta_abierto:.5f})', color='tab:blue', linewidth=2.0)
    ax2.plot(r, phi_cerrado, label=f'Teórica cerrada (ζ={zeta_cerrado:.5f})', color='tab:orange', linewidth=2.0, linestyle='--')

    # Transformación de fases experimentales al intervalo [0, 180]
    phi_exp_ab = np.array([-132.0263, -261.3675, -197.9133])
    phi_exp_ce = np.array([-260.7307,   -4.0398,    0.2567])
    def to_0_180(phi_deg):
        phi_mod = np.mod(phi_deg, 360.0)
        return np.where(phi_mod > 180.0, 360.0 - phi_mod, phi_mod)
    phi_exp_ab_0180 = to_0_180(phi_exp_ab)
    phi_exp_ce_0180 = to_0_180(phi_exp_ce)

    ax2.scatter(r_exp_cerrada, phi_exp_ce_0180, color='purple', marker='s', edgecolor='black', linewidths=0.8,
                zorder=5, s=56, alpha=0.95, label='Exp. cerrada')
    ax2.scatter(r_exp_abierta, phi_exp_ab_0180, facecolors='none', edgecolors='red', marker='o', linewidths=1.2,
                zorder=6, s=70, label='Exp. abierta')

    ax2.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlim(0.4, 1.6)
    ax2.set_ylim(0, 180)
    ax2.set_xlabel(r'$\omega_f / \omega_n$')
    ax2.set_ylabel(r'$\varphi\ (^{\circ})$')
    ax2.set_title('Desfase en función de la razón de frecuencias')
    ax2.grid(True, linestyle='--', alpha=0.25)
    ax2.legend(loc='upper right')

    output_path2 = 'fase_vs_r_abierta_cerrada.png'
    fig2.subplots_adjust(top=0.92, bottom=0.1, left=0.09, right=0.98)
    fig2.savefig(output_path2, dpi=300)
    print(f'Figura guardada en: {output_path2}')

if __name__ == '__main__':
    main()