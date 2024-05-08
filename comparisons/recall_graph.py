import matplotlib.pyplot as plt
import numpy as np
import random

# Generate dummy data
def recall_with_dist():
    distances = np.linspace(0.5, 5.0, 10)
    recalls = [0.95]

    # Generate recalls with gradually decreasing trend
    for i in range(1, len(distances)):
        prev_recall = recalls[-1]
        deviation = random.uniform(-0.1, 0.1)
        new_recall = max(0.5, prev_recall - 0.05 + deviation)
        recalls.append(new_recall)

    # Plot the graph
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(distances, recalls, marker='o')

    ax.set_xlabel('Distance of Face from Camera (meters)')
    ax.set_ylabel('Recall')
    ax.set_title('Distance vs. Recall')

    plt.show()
    
def recall_with_face_angle():
    angles = np.linspace(0, 90, 100)
    recall = np.cos(np.radians(angles))  # Recall decreases with increasing angle

    # Plot the graph
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(angles, recall)

    ax.set_xlabel('Angle of Faces (degrees)')
    ax.set_ylabel('Recall')
    ax.set_title('Angle of Faces vs. Recall')

    plt.show()

recall_with_face_angle()