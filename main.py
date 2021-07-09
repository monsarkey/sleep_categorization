import pandas as pd
import numpy as np
import os
import pyedflib # library to parse European Data Format (EDF)
from parse_RR import DaySleep


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
            day_sleep.data = pyedflib.highlevel.read_edf(f"{dirname}{filename}", ch_nrs=[3])[0][0]
        elif filename.endswith("Hypnogram.edf"):
            day_sleep.parse_labels(pyedflib.highlevel.read_edf(f"{dirname}{filename}")[2]['annotations'])

        if index % 2 != 0:
            day_sleep.split_epochs()
            data_list.append(day_sleep)

        if index == 11:
            print(f"stopping on day {index}")
            break

    return data_list

load_data()