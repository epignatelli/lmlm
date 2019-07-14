from abc import ABC, abstractmethod
import numpy as np
import random
import torch
import tensorflow as tf


class AbstractDataset(ABC):
    '''
    Defines the abstract constructor that can be used in the implemented classes
    Note that you cannot create an instance of AbstracDataset
    '''
    def __init__(self, samples=None, transform=None, **kwargs):
        self.samples = [samples if samples is not None else self.samples()]
        self.transform = transform

        for k, v in kwargs.items():
            self.k = v
        return

    # Override methods
    # ---
    def __getitem__(self, i):
        """
        Reading samples on the fly can be a heavy task, so be careful with this.
        If yo uexpect the read_sample method to be computationally expensive,
        you can consider using lmlm.data.AbstractQueue to asynchroneously prefetch your data
        """
        point = self.read_sample(self.samples[i])
        if self.transform:
            point = self.transform(point)

        # at this point, we are assuming that `getattr(point, "input") and getattr(point ,"output") == True`
        # which means that either self.samples[i] or read_sample(self.samples[i]) have those properties
        return (point.input, point.output)

    def __len__(self):
        """
        The default implementation returns len(self.samples)
        """
        return len(self.samples)

    # Abstract methods
    # ---
    @abstractmethod
    def samples(self, *args, **kwargs):
        """
        Returns a list of objects that are processed by the read_sample method.
        The implementation of this method is mandatory
        :returns: a list of objects that contains all the samples to read
        """
        raise NotImplementedError

    # Public properties
    # ---
    @property
    def input_shape(self):
        """
        Returns the input shape of the dataset.
        The default implementation assumes that self.samples[i] has the attribute "input"
        and that its values has a property "shape"
        """
        if self.samples and len(self.samples) > 0:
            return self.samples[0].input.shape

    # Compute methods
    # ---
    def read_sample(self, sample, *args, **kwargs):
        """
        Converts the sample into an object with meaning in the target framework
        e.g. you can convert the sample to a numpy array
        The default implementation the sample itself with no checks, asssuming it requires no conversion
        """
        return sample

    # Convert methods
    # ---
    def to_torch(self, point, **kwargs):
        """
        Returns the torch.tensor of the point instance
        :params point: an object that can be fed to the torch.tensor method
        :returns: a torch.tensor object
        """
        return torch.tensor(point)

    def to_tftensor(self, point, **kwargs):
        """
        Returns the tf.tensor of the point instance
        :params point: an object that can be fed to the tf.convert_to_tensor method
        :returns: a tf.tensor object
        """
        return tf.convert_to_tensor(point, **kwargs)

    def to_numpy(self, point, **kwargs):
        """
        Returns the tf.tensor of the point instance
        :params point: an object that can be fed to the np.array method
        :returns: a numpy.ndarray object
        """
        return np.array(point, **kwargs)

    def to_json(self, idx):
        """
        Returns a dicionary or json representation of the objects
        The default implemntation returns self.__dict__
        :params idx: the index of the sample in the self.samples list
        :returns: a dictionary or json objecy
        """
        return self.__dict__

    # Modify methods
    # ---

    def split(self, val_perc=0.2, test_perc=0.1):
        """
        Splits the Dataset into three Subsets: train, validation and test
        :params val_perc: a float representing the percentage of the validation subset
        :params test_perc: a float representing the percentage of the test subset
        """
        val_perc = int(len(self) * val_perc)
        test_perc = int(len(self) * test_perc)

        idx = list(range(len(self)))  # indices to all elements
        random.shuffle(idx)  # in-place shuffle the indices to facilitate random splitting

        test_idx = idx[:test_perc]
        val_idx = idx[test_perc:(test_perc + val_perc)]
        train_idx = idx[(test_perc + val_perc):]

        # Returns respectively the traning_set and the validation_set as Subsets
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)
