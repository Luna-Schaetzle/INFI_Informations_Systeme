import numpy as np
import matplotlib.pyplot as plt

rannormal = np.random.normal(0, 1, 1000)


plt.hist(rannormal, bins=30, density=True, alpha=0.6, color='g')
plt.title('Normalverteilung')
plt.xlabel('Wert')
plt.ylabel('Dichte')
plt.show()


rangleichverteilung = np.random.uniform(0, 1, 1000)

plt.hist(rangleichverteilung, bins=30, density=True, alpha=0.6, color='g')
plt.title('Gleichverteilung')
plt.xlabel('Wert')
plt.ylabel('Dichte')
plt.show()

