import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def heart_equation(x, k):
    term1 = np.abs(x)**(2/3)
    term2 = 0.9 * np.sin(k * x) * np.sqrt(np.clip(3 - x**2, 0, None))
    return term1 + term2

x = np.linspace(-np.sqrt(5), np.sqrt(5), 100000)

initial_k = 29.09
y = heart_equation(x, initial_k)

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
line, = ax.plot(x, y, color='hotpink', linewidth=2)
ax.axhline(0, color='lightblue', lw=0.5)
ax.axvline(0, color='lightblue', lw=0.5)
ax.set_title("Heart Equation (Interactive)", color='orange', fontsize=16)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')
ax.set_xticks([])
ax.set_yticks([])


slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgray')
k_slider = Slider(slider_ax, 'k', 1.0, 60.0, valinit=initial_k, valstep=0.1)


def update(val):
    
    k = k_slider.val
    line.set_ydata(heart_equation(x, k))
    fig.canvas.draw_idle()

k_slider.on_changed(update)

plt.show()
