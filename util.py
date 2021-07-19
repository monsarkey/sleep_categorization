import pandas as pd
import numpy as np
import os
import pyedflib # library to parse European Data Format (EDF)
from parse_RR import DaySleep


def parse_edf(dirname: str = "data/sleep-cassette/") -> ([DaySleep], pd.DataFrame):

    data_list = []
    tags = pd.read_csv("data/SC-subjects.csv", usecols=[2, 3])

    prev = None
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
            day_sleep.clean_data()
            # day_sleep.draw_resp(filename=filename, count=index//2, standardized=True, normalized=False, debug=True)
            data = day_sleep.get_intervals()
            # data.trim()
            if prev is not None:
                df = pd.concat([prev, data.to_df()])
            else:
                df = data.to_df()
            prev = df
            # data.draw()
            data_list.append(day_sleep)

        # if index == 11:
        #     print(f"stopping on day {index//2}")
        #     break

    return data_list, df