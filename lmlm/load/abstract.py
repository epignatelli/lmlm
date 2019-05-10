from abc import ABC, abstractmethod
import numpy as np
import random
import torch
import tensorflow as tf


class AbstractDataset(ABC):
    '''Crea il dataset di pytorch dai dati di sofifa e/o storici
    '''
    def __init__(self, **kwargs):
        self.debug = kwargs.get("debug", False)

        self.samples = self.get_samples(**kwargs)
        self.input_size = self.get_input_size()

        self.mean = None
        self.std = None

        self.set_statistics()
        return

    # *************************************************************** #
    # *** THIS METHOD HAS TO BE OVERRIDDEN IN EACH IMPLEMENTATION *** #
    # *** OF DATIPARTITE TO ENABLE YOU TO SET ANY DATA YOU WANT   *** #
    # *************************************************************** #

    @abstractmethod
    def get_samples(self, **kwargs):
        raise NotImplementedError

    # ********************************************** #

    def __getitem__(self, i):
        sample = self.samples[i]
        return
        
    # ********************************************** #

    def __len__(self):
        return len(self.samples)

    # ********************************************** #

    def get_input_size(self):
        if self.samples and len(self.samples) > 0:
            return self.samples[0]["input"].shape

    # ********************************************** #

    def set_statistics(self):
        if self.samples:
            inputs = np.array([x['input'] for x in self.samples])
            self.mean = np.mean(inputs, axis=0)
            self.std = np.std(inputs, axis=0)
            return True
        else:
            return False

    # ********************************************** #

    def fetch(self):
        # TODO: implement fetch from queue to take one sample from the queue
        pass

    # ********************************************** #

    def prefetch(self):
        # TODO: implement enqueue to keep the queue full
        pass

    # ********************************************** #

    def normalise(self, sample):
        return (sample - self.mean) / self.std

    # ********************************************** #

    def to_torch(self, sample):
        return torch.from_numpy(sample).float()

    # ********************************************** #

    def to_tftensor(self, sample):
        return tf.convert_to_tensor(sample, dtype="float32")

    # ********************************************** #

    def to_ndarray(self, sample):
        return sample.astype("float32")

    # ********************************************** #

    def to_mongo(self, idx):
        return {
                "frame": idx,
                "box": self.crop_box,
                "path": self.directory[idx],
                "locations": self.locations,
        }

    # ********************************************** #

    def split(self, val_perc=0.2, test_perc=0.1):
        val_perc = int(len(self) * val_perc)
        test_perc = int(len(self) * test_perc)

        idx = list(range(len(self)))  # indices to all elements
        random.shuffle(idx)  # in-place shuffle the indices to facilitate random splitting

        test_idx = idx[:test_perc]
        val_idx = idx[test_perc:(test_perc + val_perc)]
        train_idx = idx[(test_perc + val_perc):]

        # Returns respectively the traning_set and the validation_set as Subsets
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)
