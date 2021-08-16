from load_data import edf_to_csv
import pandas as pd
from keras_train import keras_train
from torch_train import torch_train

trimmed = True
trimmed_str = "_trimmed" if trimmed else ""

cleaned = True
cleaned_str = "_cleaned" if cleaned else ""

normalized = True
normalized_str = "_normalized" if normalized else ""


if __name__ == '__main__':

    try:
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")
    except FileNotFoundError:
        print("file not found, reloading data from .edf")
        edf_to_csv(trimmed=trimmed, cleaned=cleaned, normalized=normalized)
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")

    # out_df, keras_model = keras_train(df)

    out_df, torch_model = torch_train(df)

    out_df.to_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}_output.csv")
    print('done!')
