# Exercise 2.6
import matplotlib.pyplot as plt
import numpy as np

# Menentukan rentang nilai x
x = np.linspace(-5, 5, 200)

# Menghitung persamaan matematika
y1 = 3 * x + 4
y2 = 2 * (x**2) + 1
y3 = (x**3) + 9

# Plotting ketiga garis dengan warna berbeda
plt.plot(x, y1, color="blue", label="y = 3x + 4")
plt.plot(x, y2, color="red", linestyle="--", label="y = 2x^2 + 1")
plt.plot(x, y3, color="green", linestyle="-.", label="y = x^3 + 9")

# Pengaturan judul, label sumbu, grid, dan legenda
plt.title("Plot of Math Functions")
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True)
plt.legend()

plt.show()