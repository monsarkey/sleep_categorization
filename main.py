from util import edf_to_csv
import pandas as pd

trimmed = True
trimmed_str = "_trimmed" if trimmed else ""

if __name__ == '__main__':

    # try:
    #     df = pd.read_csv(f"data/edf_data{trimmed_str}.csv")
    # except FileNotFoundError:
    #     print("file not found, reloading data from .edf")
    #     edf_to_csv(trimmed=trimmed)
    #     df = pd.read_csv(f"data/edf_data{trimmed_str}.csv")

    edf_to_csv(trimmed=trimmed)
    # print(df)