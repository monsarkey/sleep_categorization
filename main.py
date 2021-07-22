from load_data import edf_to_csv
import pandas as pd
from model import CNN1D

trimmed = False
trimmed_str = "_trimmed" if trimmed else ""

cleaned = True
cleaned_str = "_cleaned" if cleaned else ""

normalized = True
normalized_str = "_normalized" if cleaned else ""

if __name__ == '__main__':

    try:
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")
    except FileNotFoundError:
        print("file not found, reloading data from .edf")
        edf_to_csv(trimmed=trimmed, cleaned=cleaned, normalized=normalized)
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")

    cnn = CNN1D((1, 10))

    # print(df)