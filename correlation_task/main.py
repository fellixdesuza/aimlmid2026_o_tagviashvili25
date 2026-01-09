import numpy as np
import matplotlib.pyplot as plt

# Given data
x = np.array([-9.70, -7.00, -4.50, -1.80, 0.70, 3.30, 5.80, 8.40])
y = np.array([-7.90, -5.40, -2.80, -0.60, 1.50, 3.60, 5.80, 8.00])

# Calculate Pearson correlation
r = np.corrcoef(x, y)[0, 1]
print("Pearson correlation (r):", r)

# Visualization
plt.figure(figsize=(7, 5))
plt.scatter(x, y)
plt.plot(x, y)  # simple line to show linear trend
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title(f"Correlation Visualization (r = {r:.4f})")
plt.grid(True)
plt.show()

