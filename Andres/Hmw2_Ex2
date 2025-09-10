import numpy as np
import matplotlib.pyplot as plt

def graph(x):
    return 1.5*x - 1.5*x**2 + 0.5*x**3

x = np.linspace(-2, 4, 1000)
y = graph(x)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, linewidth=2, color='tab:red', label=r'Normalized Load – Deflection')

ax.set_xlabel(r'$-\frac{w}{z}$', fontsize=14)
ax.set_ylabel(r'$-\frac{W}{EA}\left(\frac{L}{z}\right)^3$', fontsize=14)

ax.text(
    0.25, 0.25,
    r'$-\frac{W}{EA}\left(\frac{L}{z}\right)^3 = -\frac{3}{2}\left(\frac{w}{z}\right) - \frac{3}{2}\left(\frac{w}{z}\right)^2 - \frac{1}{2}\left(\frac{w}{z}\right)^3$',
    transform=ax.transAxes, fontsize=14,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

ax.set_xlim(-2, 4)
ax.set_ylim(-4, 4)
ax.minorticks_on()
ax.grid(True, which='both', linestyle=':', linewidth=0.8)
ax.legend(frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
plt.show()
