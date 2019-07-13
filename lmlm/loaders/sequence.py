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
import matplotlib.pyplot as plt
import random as rn


class DataSequence(object):
    def __init__(self,
                 directory,
                 locations,
                 crop_box,
                 mongo_collection=None,
                 mean=None,
                 fps=1):

        self.directory = directory
        self.locations = locations
        self.crop_box = crop_box
        self.fps = fps

        self.frames = self.get_frames(directory)

        self.set_collection(mongo_collection)
        self.set_mean(mean)
        
        self.print_random()
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

    def get_batch(self, idx):
        start = time.time()
        print("Fetching batch %s" % idx, end="\r")
        frame = self.frames[idx]
        size = self.mean.shape if self.mean else None
        people_dic = Compute.extract.people(frame,
                                            self.locations,
                                            self.crop_box,
                                            size=size)
        people_img, people_loc = people_dic["array"], people_dic["locations"]
        people_img = np.stack(people_img)
        people_img = self.normalise(people_img)
        if self.mongo_collection is not None:
            self.mongo_collection.insert_one(self.to_mongo(idx))
        print("Fetched in %4f" % (time.time() - start), end="\r")
        return people_img, people_loc

    # ********************************************** #

    def get_input_size(self):
        return (self.crop_box[1] - self.crop_box[0], self.crop_box[3] - self.crop_box[2], 3)

    # ********************************************** #

    def set_mean(self, mean=None):
        if mean is None:
            try:
                collection = pymongo.MongoClient()["safestanding"]["statistics"]
                mean = collection.find().sort("date", -1)[0]["mean"]
                mean = np.array(mean)
            except:
                self.mean = None
                traceback.print_exc()
                return
        self.mean = np.swapaxes(mean, 0, 1)
        return

    # ********************************************** #

    def set_collection(self, mongo_collection):
        if mongo_collection is "auto":
            stadium, match, camera = Path(self.directory).parts[-4:-1]
            collection_name = "_".join([stadium, match, camera])
            i = 1
            db = pymongo.MongoClient()["safestanding_data"]
            while db[collection_name].count() > 0:
                collection_name = collection_name + str(i)
                i += 1
            self.mongo_collection = db[collection_name]
        elif mongo_collection == "test":
            self.mongo_collection = pymongo.MongoClient()["development"]["test"]
        else:
            self.mongo_collection = mongo_collection
        return

    # ********************************************** #

    def set_global_statistics(self):
        if self.frames:
            self.mean = Query.stats.mean(self.frames, self.get_input_size())
        return

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
    
    def unnormalise(self, sample):
        sample = sample * 255
        if self.mean is not None:
            sample = sample + self.mean
        return sample

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
                "path": self.frames[idx],
                "locations": self.locations,
        }

    # ********************************************** #
    
    def print_random(self):
        idx = rn.randint(0, len(self))
        sample = self.get_batch(idx)[0]
        idx_b = rn.randint(0, len(sample))
        print("Example batch: [%d][%d]" % (idx, idx_b))
        sample = sample[idx_b]
        print(sample)
        plt.imshow(self.unnormalise(sample).astype("int32"))
        return
    
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
