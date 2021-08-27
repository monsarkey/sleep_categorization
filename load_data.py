import pandas as pd
import os
import pyedflib # library to parse European Data Format (EDF)
from parse_RR import DaySleep

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: July 22nd, 2021

This file handles the flow for loading data from .edf, parsing it into respiratory epochs, trimming into a
night's sleep, and then compiling into .csv as training data
"""

# convert from edf to a single dataframe containing training data
def parse_edf(dirname: str = "data/sleep-cassette/", trimmed: bool = False,
              cleaned: bool = True, normalized: bool = False) -> ([DaySleep], pd.DataFrame):

    data_list = []
    prev = None

    tags = pd.read_csv("data/SC-subjects.csv", usecols=[2, 3])

    # store our .edf files together in {dirname} directory to be used here
    for index, filename in enumerate(os.listdir(dirname)):

        # compile data from sleep cassette into DaySleep object
        if index % 2 == 0:
            day_sleep = DaySleep()
        else:
            day_sleep.age = tags["age"][index // 2]
            day_sleep.gender = tags["sex (F=1)"][index // 2]

        if filename.endswith("PSG.edf"):
            day_sleep.data = pyedflib.highlevel.read_edf(f"{dirname}{filename}", ch_nrs=[3])[0][0]
        elif filename.endswith("Hypnogram.edf"):
            day_sleep.parse_labels(pyedflib.highlevel.read_edf(f"{dirname}{filename}")[2]['annotations'])

        # process data inside of new DaySleep class
        if index % 2 != 0:
            day_sleep.split_epochs()

            # if we want, remove outliers from data
            if cleaned:
                day_sleep.clean_data()

            data = day_sleep.get_intervals(trimmed=trimmed, normalized=normalized)

            # concatenate current dataframe onto previous
            if prev is not None:
                df = pd.concat([prev, data.to_df()], ignore_index=True)
            else:
                df = data.to_df()
            prev = df
            data_list.append(day_sleep)

        if index % 10 == 1:
            print(f"finished processing day {index//2 + 1}...")

    return data_list, df


# take the dataframe returned by parse_edf and turn it into appropriately labeled .csv
def edf_to_csv(trimmed: bool = False, cleaned: bool = True, normalized: bool = False):
    _, df = parse_edf(trimmed=trimmed, cleaned=cleaned, normalized=normalized)

    filepath = "data/edf_data_trimmed" if trimmed else 'data/edf_data'
    if cleaned:
        filepath += "_cleaned"
    if normalized:
        filepath += "_normalized"

    df.to_csv(f"{filepath}.csv")
