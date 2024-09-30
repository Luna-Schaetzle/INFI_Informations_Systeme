import numpy as np

normalverteilte_zahlen = np.random.normal(size=100 * 100)

print([normalverteilte_zahlen[i * 100:(i + 1) * 100].mean() for i in range(100)])

