import pandas as pd
import numpy as np
import os
import pyedflib # library to parse European Data Format (EDF)

class DaySleep:

    def __init__(self, age: int = None, gender: int = None, data: np.ndarray = None, labels: np.ndarray = None):
        self.age = age
        self.gender = gender
        self.data = data
        self.labels = labels

    # def add_data(self, data):
    #     self.data = data
    #
    # def add_labels(self, labels):
    #     self.labels = labels


def load_data(dirname: str = "data/sleep-cassette/") -> [DaySleep]:

    data_list = []
    tags = pd.read_csv("data/SC-subjects.csv", usecols=[2, 3])

    for index, filename in enumerate(os.listdir(dirname)):

        if index % 2 == 0:
            day_sleep = DaySleep()
        else:
            day_sleep.age = tags["age"][index // 2]
            day_sleep.gender = tags["sex (F=1)"][index // 2]

        if filename.endswith("PSG.edf"):
            day_sleep.data = pyedflib.highlevel.read_edf(f"{dirname}{filename}", ch_nrs=[3])[0]
        elif filename.endswith("Hypnogram.edf"):
            day_sleep.labels = pyedflib.highlevel.read_edf(f"{dirname}{filename}")[2]['annotations']

        if index % 2 != 0:
            data_list.append(day_sleep)

    return data_list

load_data()