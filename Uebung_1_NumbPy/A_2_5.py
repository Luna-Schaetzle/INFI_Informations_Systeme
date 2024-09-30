import numpy as np

rannormal = np.random.normal(0, 1, 1000)

# alle zahlen größer als 0 in einem Array ausgeben

positivenormal = rannormal[rannormal > 0]
print(positivenormal)
