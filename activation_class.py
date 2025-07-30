import numpy as np




class ReLU:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)


