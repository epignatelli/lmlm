import numpy as np
import torch
import tensorflow as tf
from abc import ABC


class DataPoint(ABC):
    def __init__(self, a, b=[], *args, **kwargs):
        if not isinstance(a, np.ndarray):
            raise TypeError("DataPoint.input must be of type numpy.ndarray")
        self.input = a
        self.output = b

    def to_torch(self, **kwargs):
        return torch.array(self.input, **kwargs)

    def to_tftensor(self, **kwargs):
        return tf.convert_to_tensor(self.input, **kwargs)
