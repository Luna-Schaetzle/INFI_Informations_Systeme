import numpy as np
import matplotlib.pyplot as plt

# Number of dice rolls
n_rolls = 1000

# Simulate dice rolls
rolls = np.random.randint(1, 7, size=n_rolls)

# Plot the results
plt.hist(rolls, bins=np.arange(1, 8) - 0.5, edgecolor='black')
plt.xticks(np.arange(1, 7))
plt.xlabel('Dice Value')
plt.ylabel('Frequency')
plt.title('Dice Roll Experiment')
plt.show()
