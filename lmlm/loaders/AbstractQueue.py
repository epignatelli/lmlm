import os
import traceback
from pathlib import Path
import random
import numpy as np
import pymongo
import torch
import tensorflow as tf
import time
import random as rn
from asyncio import Queue
from . import AbstractDataset
from . import Subset
from abc import ABC


class AbstractQueue(AbstractDataset, ABC):
    def __init__(self, maxsize=3, *args):
        super().__init__(*args)

        # initialise queue
        self.queue = Queue(maxsize=maxsize)
        return

    # ********************************************** #

    def __iter__(self):
        # self.background_worker = Thread(target=self.enqueue(), daemon=True)
        # self.background_worker.start()
        return self

    # ********************************************** #

    def __next__(self):
        if not self.queue.empty():
            batch = self.queue.get()
            self.queue.task_done()
            return batch
        else:
            raise StopIteration

    async def request_sample(self, idx):
        start = time.time()
        print("Fetching new sample...", end="\r")
        sample = await self.samples[idx]
        print("Fetched in %4f" % (time.time() - start))
        return sample

    def _producer(self):
        # this is async
        for frame in self.frames:
            t = Thread(target=self.request_sample, args=frame)
            self.queue.put(t)
        self.queue.join()
        return

    def print_random(self):
        idx = rn.randint(0, len(self))
        sample = self.request_sample(idx)
        print(f"Example batch {idx:s}")
        print(sample)

    # def __del__(self):
    #     if not hasattr(self, "background_worker"):
    #         super().__del__()
    #     elif self.background_worker is not None:
    #         self.background_worker.join()
    #         self.background_worker.stop()
    #     return

# class Worker(Thread):
#     def __init__(self, target, daemon):
#         super().__init(target=target, daemon=daemon)
#         self.alive = threading.Event()
#         self.alive.set()

#     def run(self):
#         while self.alive.isSet():
#             try:
