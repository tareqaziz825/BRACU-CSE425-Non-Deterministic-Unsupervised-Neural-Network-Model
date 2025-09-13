# Self-Organizing Map implementation

import numpy as np

class SOM:
    def __init__(self, grid_size, input_dim, learning_rate, n_iterations):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)

    def find_bmu(self, x):
        distances = np.sum((self.weights - x)**2, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_weights(self, x, bmu_index, learning_rate, neighborhood_radius):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                neuron_distance = np.sqrt((i - bmu_index[0])**2 + (j - bmu_index[1])**2)
                if neuron_distance <= neighborhood_radius:
                    influence = np.exp(-neuron_distance**2 / (2 * neighborhood_radius**2))
                    self.weights[i, j, :] += learning_rate * influence * (x - self.weights[i, j, :])

    def train(self, X):
        initial_learning_rate = self.learning_rate
        initial_neighborhood_radius = max(self.grid_size) / 2.0
        for iteration in range(self.n_iterations):
            learning_rate = initial_learning_rate * (1 - iteration / self.n_iterations)
            neighborhood_radius = initial_neighborhood_radius * np.exp(-iteration / self.n_iterations)
            for x in X:
                bmu_index = self.find_bmu(x)
                self.update_weights(x, bmu_index, learning_rate, neighborhood_radius)

    def cluster(self, X):
        labels = []
        for x in X:
            bmu_index = self.find_bmu(x)
            labels.append(bmu_index[0] * self.grid_size[1] + bmu_index[1])
        return np.array(labels)
