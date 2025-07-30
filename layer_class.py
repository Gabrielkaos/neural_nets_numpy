import numpy as np





class Layer:
    def __init__(self, n_features, n_out):
        self.weights = np.random.randn(n_features, n_out) * 0.05
        self.biases = np.zeros((1,n_out))
        self.outputs = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases




layer1 = Layer(2,3)
layer2 = Layer(3,1)

inputs = np.array([[0.1,0.2],
                   [0.3,0.1],
                   [0.2,0.2]])

layer1.forward(inputs)
print(layer1.outputs)
layer2.forward(layer1.outputs)
print(layer2.outputs)