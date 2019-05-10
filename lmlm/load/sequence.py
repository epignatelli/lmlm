import os
import traceback
from pathlib import Path
import random
import numpy as np
import pymongo
import torch
import tensorflow as tf
from Engine import Compute
from Engine import Query
from datasets.subset import Subset
import time


class Sequence(object):
    def __init__(self,
                 directory,
                 locations,
                 crop_box,
                 host="127.0.0.1",
                 mean=None,
                 fps=1000):

        self.directory = directory
        self.locations = locations
        self.crop_box = crop_box
        self.host = host
        self.fps = fps

        self.frames = self.get_frames(directory)

        self.database = self.get_database()

        self.mean = self.try_get_mean(mean)

        print("Example batch")
        print(self.get_batch(0))
        return

    # ********************************************** #

    def __getitem__(self, idx):
        return self.get_batch(idx)

    # ********************************************** #

    def __len__(self):
        return len(self.frames)

    # ********************************************** #

    def get_frames(self, directory):
        print("Getting files from %s" % directory)
        names = os.listdir(directory)
        paths = []
        for i in range(0, len(names), self.fps):
            path = os.path.join(directory, names[i])
            if self.is_image(path):
                paths.append(path)
        print("%d files found in %s" % (len(paths), directory))
        return paths

    # ********************************************** #

    def get_input_size(self):
        return (self.crop_box[1] - self.crop_box[0], self.crop_box[3] - self.crop_box[2])

    # ********************************************** #

    def try_get_mean(self, mean=None):
        if mean is not None:
            mean = np.expand_dims(mean, 0)
        try:
            collection = pymongo.MongoClient()["safestanding"]["statistics"]
            mean = collection.find().sort("date", -1)[0]["mean"]
            mean = np.array(mean)
        except:
            traceback.print_exc()
            mean = None
        return np.expand_dims(mean, 0)

    # ********************************************** #

    def get_database(self):
        stadium, match, camera = Path(self.directory).parts[-4:-1]
        collection_name = "_".join([stadium, match, camera])
        return pymongo.MongoClient(self.host)["safestanding"][collection_name]

    # ********************************************** #

    def set_global_statistics(self):
        if self.frames:
            self.mean = Query.stats.mean(self.frames, self.get_input_size())
        return

    # ********************************************** #

    def get_batch(self, idx):
        start = time.time()
        print("Fetching batch %s" % idx, end="\r")
        frame = self.frames[idx]
        people_img = Compute.extract.people(frame, self.locations, self.crop_box)
        people_img = np.stack(people_img)
        people_img = np.swapaxes(people_img, 1, 2)
        people_img = self.normalise(people_img)
        self.database.insert_one(self.to_mongo(idx))
        print("Fetched in %4f" % (time.time() - start), end="\r")
        return people_img, None

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
        if self.mean is not None:
            sample = sample - self.mean
        return sample / 255

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

    @staticmethod
    def is_image(path):
        if not os.path.isfile(path):
            return False
        exts = ["jpg", "jpeg", "png"]
        for ext in exts:
            test = path.endswith(ext)
            if test:
                return True
        return False

    # ********************************************** #
