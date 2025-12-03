import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Parámetros del sistema
omega_n = 39.729  # rad/s
zeta_abierta = 0.0028
zeta_cerrada = 0.0260

# Rango para r y funciones teóricas
r = np.linspace(0.1, 1.6, 800)

def FA(r, z):
    return 1.0 / np.sqrt((1 - r**2)**2 + (2*z*r)**2)

def phi_deg(r, z):
    # φ teórico en grados
    phi_rad = np.arctan2(2*z*r, (1 - r**2))
    # Llevar a [0, 180) por claridad en la comparación
    phi_deg = np.degrees(phi_rad)
    phi_deg = (phi_deg + 360) % 360
    phi_deg = np.where(phi_deg > 180, 360 - phi_deg, phi_deg)
    return phi_deg

# Puntos experimentales (desde la tabla del .tex)
# r = omega_f/omega_n con omega_f = 2πf
freqs = np.array([4.20, 4.30, 4.45])
omega_f = 2*np.pi*freqs
r_pts = omega_f/omega_n

# FAexp de la tabla (columna FA)
FA_abierta_pts = np.array([1.7895, 1.8603, 1.9813])
FA_cerrada_pts = np.array([1.7895, 1.8603, 1.9813])  # mismos valores en la tabla provista

# φexp estilo guía (+π/2), en grados, de la tabla actualizada
phi_abierta_deg_pts = np.array([125.93, 80.55, 11.57])
phi_cerrada_deg_pts = np.array([74.19, 50.21, 10.75])

# Salidas
root = Path(r"c:\\Users\\lunit\\OneDrive\\Desktop\\laboratorio dinamica")

# Figura FA vs r
plt.figure(figsize=(7.2, 4.5), dpi=150)
plt.plot(r, FA(r, zeta_abierta), 'b-', label=f'Teórico abierta (ζ={zeta_abierta})')
plt.plot(r, FA(r, zeta_cerrada), 'g-', label=f'Teórico cerrada (ζ={zeta_cerrada})')

# Puntos experimentales (colores según pie de figura del .tex)
plt.plot(r_pts, FA_abierta_pts, 'o', mfc='none', mec='red', mew=1.5, label='Experimento abierta')
plt.plot(r_pts, FA_cerrada_pts, 'o', color='#7D3C98', label='Experimento cerrada')

plt.xlabel(r'$r = \omega_f / \omega_n$')
plt.ylabel('FA')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
(fig1_path := root / 'FA_vs_r_abierta_cerrada.png')
plt.savefig(fig1_path)
print(f'Guardado: {fig1_path}')
plt.close()

# Figura fase vs r
plt.figure(figsize=(7.2, 4.5), dpi=150)
plt.plot(r, phi_deg(r, zeta_abierta), 'b-', label=f'Teórico abierta (ζ={zeta_abierta})')
plt.plot(r, phi_deg(r, zeta_cerrada), 'g-', label=f'Teórico cerrada (ζ={zeta_cerrada})')

plt.plot(r_pts, phi_abierta_deg_pts, 'o', mfc='none', mec='red', mew=1.5, label='Experimento abierta')
plt.plot(r_pts, phi_cerrada_deg_pts, 'o', color='#7D3C98', label='Experimento cerrada')

plt.xlabel(r'$r = \omega_f / \omega_n$')
plt.ylabel(r'$\varphi$ (grados)')
plt.ylim(-5, 185)
plt.yticks(np.arange(0, 181, 30))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
(fig2_path := root / 'fase_vs_r_abierta_cerrada.png')
plt.savefig(fig2_path)
print(f'Guardado: {fig2_path}')
plt.close()
