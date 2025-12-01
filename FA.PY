import numpy as np
import matplotlib.pyplot as plt

# Datos experimentales
r_exp = [0.6642, 0.6800, 0.7038, 0.6642, 0.6800, 0.7038]
FA_exp_abierta = [0.0062, 0.0060, 0.0058]
FA_exp_cerrada = [0.0062, 0.0060, 0.0058]

# Parámetros (ajusta c_abierto y c_cerrado según tus datos)
zeta_abierto = 0.01  # Ejemplo, reemplaza por tu valor real
zeta_cerrado = 0.05  # Ejemplo, reemplaza por tu valor real

r = np.linspace(0.6, 1.2, 400)

def FA_teorico(r, zeta):
    return 1 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)

FA_abierto = FA_teorico(r, zeta_abierto)
FA_cerrado = FA_teorico(r, zeta_cerrado)

plt.figure(figsize=(8,5))
plt.plot(r, FA_abierto, label=f'Abierta  (c= {zeta_abierto} Ns/m)')
plt.plot(r, FA_cerrado, label=f'Cerrada  (c= {zeta_cerrado} Ns/m)')
plt.scatter(r_exp[:3], FA_exp_abierta, color='red')
plt.scatter(r_exp[3:], FA_exp_cerrada, color='red')
plt.axvline(1, color='gray', linestyle='--', alpha=0.7)
plt.xlabel(r'$\omega_f / \omega_n$')
plt.ylabel('FA')
plt.title('FA – amortiguador abierto vs cerrado')
plt.legend()
plt.tight_layout()
plt.show()