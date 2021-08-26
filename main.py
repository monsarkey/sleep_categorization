import pandas as pd

from keras_train import keras_train
from load_data import edf_to_csv
from torch_train import torch_train
from analysis import visualize_PCA, visualize_LDA, draw_vars_1D

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: July 8th 2021

The purpose of this project effectively to categorize sleep stages into awake/light/deep/rem using solely
respiratory data. In order to do this, we use the Expanded Sleep-EDF Dataset, which includes recordings of
inner-nasal voltage from which respiratory rate and strength can be estimated. We perform manual feature extraction
on this data and concatenate features from every night's sleep into one dataset. We then use this dataset to feed
into a machine learning model for classification.

This is the main file of my sleep categorization work. It handles flow of the entire program, through the
parsing of .edf files to their analysis and visualization, and then their fitting to models in either keras
or pytorch. This file also allows for the modification of important parameters regarding the data (whether it
should be trimmed, cleaned, and normalized) and choose between models to use for fitting. 
"""


# whether or not we should trim the dataset to only nighttime, or include all 24hrs
trimmed = True
trimmed_str = "_trimmed" if trimmed else ""

# whether or not we should remove and interpolate outliers in the data
cleaned = True
cleaned_str = "_cleaned" if cleaned else ""

# whether or not we should normalize the features of the data
normalized = True
normalized_str = "_normalized" if normalized else ""

# whether or not we should perform data analysis and visualization
analysis = False

# choose either "pytorch" or "keras"
train_method = "pytorch"


if __name__ == '__main__':

    # attempt to read in already parsed data from .csv
    try:
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")
    except FileNotFoundError:
        print("file not found, reloading data from .edf")

        # if we don't already have parsed data, parse it here according to parameters above
        edf_to_csv(trimmed=trimmed, cleaned=cleaned, normalized=normalized)
        df = pd.read_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}.csv")

    if analysis:
        # 3D visualizations of Principal Component Analysis and Linear Discriminant Analysis
        visualize_PCA(df, nr_dim=3, frac=.5)
        visualize_LDA(df, frac=.05, standardize=False)
        # 1D graphs showing correlations of individual variables
        draw_vars_1D(df, frac=.005, standardize=False)

    if train_method == "pytorch":
        out_df, torch_model = torch_train(df)
    elif train_method == "keras":
        out_df, keras_model = keras_train(df)
        out_df.to_csv(f"data/edf_data{trimmed_str}{cleaned_str}{normalized_str}_output.csv")

    print('done!')
