import numpy as np

normalverteilte_zahlen = np.random.normal(size=100)

unteres_quantil = np.percentile(normalverteilte_zahlen, 2.5)
oberes_quantil = np.percentile(normalverteilte_zahlen, 97.5)

innerhalb_95 = normalverteilte_zahlen[(normalverteilte_zahlen >= unteres_quantil) & (normalverteilte_zahlen <= oberes_quantil)]
außerhalb_95 = normalverteilte_zahlen[(normalverteilte_zahlen < unteres_quantil) | (normalverteilte_zahlen > oberes_quantil)]

print("Unteres Quantil (2.5%):", unteres_quantil)
print("Oberes Quantil (97.5%):", oberes_quantil)
print("Werte innerhalb der mittleren 95%:", innerhalb_95)
print("Werte außerhalb der mittleren 95%:", außerhalb_95)