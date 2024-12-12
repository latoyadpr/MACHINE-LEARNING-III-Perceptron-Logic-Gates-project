import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Inputs to AND
data = [[0, 0], [0, 1], [1, 0], [1, 1]]


# Labels for AND
labels = [0, 0, 0, 1]


# Plotting the points
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)

# Display the

classifier = Perceptron(max_iter=40, random_state=22)


classifier = Perceptron(max_iter=40, random_state=22)

classifier.fit(data, labels)


accuracy = classifier.score(data, labels)
print(accuracy)


# Labels for XOR
labels = [0, 1, 1, 0]


classifier.fit(data, labels)
accuracy = classifier.score(data, labels)
print(accuracy)


# Labels for OR
labels = [0, 1, 1, 1]


classifier.fit(data, labels)
accuracy = classifier.score(data, labels)
print(accuracy)

# Labels for AND
labels = [0, 0, 0, 1]

classifier.fit(data, labels)

distances = classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]])
print(distances)

x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)


from itertools import product

point_grid = list(product(x_values, y_values))


distances = classifier.decision_function(point_grid)

abs_distances = [abs(distance) for distance in distances]

lst = [1, 2 ,3, 4]
new_lst = np.reshape(lst, (2, 2))


# Reshape abs_distances into a 100 by 100 matrix
distances_matrix = np.reshape(abs_distances, (100, 100))

# Create the heat map
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)

# Add a color bar
plt.colorbar(heatmap)

# Display the plot
plt.show()


# Labels for XOR
labels = [0, 1, 1, 0]

# Retrain the perceptron
classifier.fit(data, labels)

# Calculate distances for the XOR gate
distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))

# Create the heat map for XOR gate
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show()
