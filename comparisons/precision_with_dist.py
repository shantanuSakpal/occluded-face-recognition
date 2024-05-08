import matplotlib.pyplot as plt
import numpy as np
import random

# Generate dummy data
distances = np.linspace(0.5, 5.0, 10)
precisions = [0.95]

# Generate precisions with small deviations
for i in range(1, len(distances)):
    prev_precision = precisions[-1]
    deviation = random.uniform(-0.05, 0.05)
    new_precision = max(0.8, min(0.98, prev_precision + deviation))
    precisions.append(new_precision)

# Plot the graph
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(distances, precisions, marker='o')

ax.set_xlabel('Distance of Face from Camera (meters)')
ax.set_ylabel('Precision')
ax.set_title('Distance vs. Precision')
# Set y-axis range from 0.5 to 1
ax.set_ylim(0.5, 1)
plt.show()