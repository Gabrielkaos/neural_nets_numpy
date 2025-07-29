import numpy as np


inputs = np.array([[0.1,0.2],
                   [0.3,0.1],
                   [0.2,0.2]])
weights = np.array([[1.0, 1.2],[2.0, 0.9]])
bias = np.array([[1.0, 3.0]])

print(inputs.shape)
print(weights.shape)
print(bias.shape)

print()
output = np.dot(inputs,weights) + bias
print(output)