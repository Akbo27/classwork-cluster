import numpy as np
import csv
import matplotlib.pyplot as plt

file_path = 'points.csv'
points= []

with open(file_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        points.append([float(row[0]), float(row[1])])

points = np.array(points)

x_min, y_min = points.min(axis=0)
x_max, y_max = points.max(axis=0)

centroids = np.array([
    [x_min, y_min],
    [x_max, y_max],
    [(x_min + x_max) / 2, (y_min + y_max) / 2]
])

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Points')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x', s=100, label='Centroids')
plt.axvline(x=x_min, color='green', linestyle='--', label='X Bounds')
plt.axvline(x=x_max, color='green', linestyle='--')
plt.axhline(y=y_min, color='purple', linestyle='--', label='Y Bounds')
plt.axhline(y=y_max, color='purple', linestyle='--')
plt.title('Points and Estimated Centroids')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()

