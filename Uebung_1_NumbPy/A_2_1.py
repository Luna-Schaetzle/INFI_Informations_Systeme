import numpy as np

normalverteilte_zahlen = np.random.normal(size=100)
mittelwert = np.mean(normalverteilte_zahlen)
median = np.median(normalverteilte_zahlen)
maximum = np.max(normalverteilte_zahlen)
minimum = np.min(normalverteilte_zahlen)
standardabweichung = np.std(normalverteilte_zahlen)

print("Normalverteilte Zahlen:")
print(normalverteilte_zahlen)
print("Mittelwert: ", mittelwert)
print("Median: ", median)
print("Maximum: ", maximum)
print("Minimum: ", minimum)
print("Standardabweichung: ", standardabweichung)




