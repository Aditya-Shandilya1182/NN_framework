#import numpy as np
from .module import Module
from src.tensor import Tensor  # Ensure you have the correct import path for Tensor

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if isinstance(x, Tensor):
            z = x.sigmoid()
            return z
        else:
            raise TypeError("Expected input to be of type Tensor")
