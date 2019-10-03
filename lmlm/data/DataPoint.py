import numpy as np
import torch
import tensorflow as tf
import json
from abc import ABC


class DataPoint(ABC):
    def __init__(self, a, b=[], *args, **kwargs):
        self.input = a
        self.output = b

    def __repr__(self):
        return (f"\{input: {self.input}; output: {self.output}\}")

    def to_torch(self, **kwargs):
        """
        Returns the torch.tensor of the point instance
        :params point: an object that can be fed to the torch.tensor method
        :returns: a torch.tensor object
        """
        in_torch = torch.tensor(self.input, **kwargs)
        out_torch = torch.tensor(self.output, **kwargs) if self.output != [] else None
        return (in_torch, out_torch)

    def to_tftensor(self, **kwargs):
        """
        Returns the tf.tensor of the point instance
        :params point: an object that can be fed to the tf.convert_to_tensor method
        :returns: a tf.tensor object
        """
        return tf.convert_to_tensor(point, **kwargs)

    def to_numpy(self, **kwargs):
        """
        Returns the tf.tensor of the point instance
        :params point: an object that can be fed to the np.array method
        :returns: a numpy.ndarray object
        """
        return np.array(point, **kwargs)

    def to_json(self, **kwargs):
        """
        Returns a dicionary or json representation of the objects
        The default implemntation returns self.__dict__
        :params idx: the index of the sample in the self.samples list
        :returns: a dictionary or json objecy
        """
        return json.dumps(self.__dict__, **kwargs)
