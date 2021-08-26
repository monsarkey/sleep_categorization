import pandas as pd

from keras_train import keras_train
from load_data import edf_to_csv
from torch_train import torch_train
from analysis import visualize_PCA, visualize_LDA, draw_vars_1D

trimmed = True
trimmed_str = "_trimmed" if trimmed else ""

cleaned = True
cleaned_str = "_cleaned" if cleaned else ""

normalized = True
normalized_str = "_normalized" if normalized else ""

analysis = False
train_method = "keras"


if __name__ == '__main__':

    try:
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")
    except FileNotFoundError:
        print("file not found, reloading data from .edf")

        edf_to_csv(trimmed=trimmed, cleaned=cleaned, normalized=normalized)
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")

    if analysis:
        visualize_PCA(df, nr_dim=3, frac=.5)
        visualize_LDA(df, frac=.05, standardize=False)
        draw_vars_1D(df, frac=.005, standardize=False)

    if train_method == "pytorch":
        out_df, torch_model = torch_train(df)
    elif train_method == "keras":
        out_df, keras_model = keras_train(df)
        out_df.to_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}_output.csv")

    print('done!')
