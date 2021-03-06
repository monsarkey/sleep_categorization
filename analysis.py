import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from util import draw_conf

"""
Author: Sean Markey (00smarkey@gmail.com)

Created: August 19th, 2021

This file contains functions used to apply dimensionality reduction and visualization to our manually selected
feature dataset, used to get a better understanding of our data. 
"""

# utility function that splits features from labels and then standardizes the features
def feature_split(df: pd.DataFrame, standardize: bool = True) -> (np.ndarray, np.ndarray):

    features = ['rr_mean', 'rr_std', 'rs_std', 'rr_range', 'rr_delta_abs',
                'rs_delta_abs', 'rr_disp', 'rr_trend', 'gender', 'age']

    x = df.reindex(columns=features).values
    y = df.reindex(columns=['label']).values.flatten()

    if standardize:
        x = StandardScaler().fit_transform(x)

    return x, y

# draws a 1D representation of all of our variables side by side to try and understand correlations
def draw_vars_1D(df: pd.DataFrame, frac: float = 1.0, standardize: bool = True):

    df = df.sample(frac=frac)

    x, y = feature_split(df, standardize=standardize)
    y = [{'awake': 0, 'light': 1, 'deep': 2, 'rem': 3}[key] for key in y]

    features = ['rr_mean', 'rr_std', 'rs_std', 'rr_range', 'rr_delta_abs',
                'rs_delta_abs', 'rr_disp', 'rr_trend', 'gender', 'age']

    df = pd.DataFrame(x, columns=features)
    colors = ['c', 'b', 'm', 'r']

    fig, axs = plt.subplots(len(features))
    fig.suptitle(f"Features Plotted Using {frac * 100:.1f}% of Data")
    for i, ax in enumerate(axs):
        ax.scatter(df[features[i]], np.zeros(df.shape[0]), c=y, cmap=matplotlib.colors.ListedColormap(colors))
        ax.set_ylabel(features[i], rotation=-35, labelpad=30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.yaxis.set_label_position("right")
    plt.savefig(f"figures/vars_1D/vars_1D_frac={frac:.3f}.png")
    plt.close()

# draws a 3D representation of Linear Discriminant Analysis applied to our dataset
def visualize_LDA(df: pd.DataFrame, frac: float = 1.0, standardize: bool = True):

    df = df.sample(frac=frac)
    x, y = feature_split(df, standardize=standardize)

    lda = LinearDiscriminantAnalysis(n_components=3).fit(x, y)
    x_r_lda = lda.transform(x)

    # check accuracy of categorization by LDA
    out = lda.predict(x)
    print(f"LDA accuracy: {lda.score(x,y)*100:.2f}%")
    draw_conf(out, y, name=f"LDA_conf_frac={frac:.2f}.png")

    ld_df = pd.DataFrame(data=x_r_lda, columns=['LD 1', 'LD 2', 'LD 3'])

    fig = px.scatter_3d(x=ld_df['LD 1'].values, y=ld_df['LD 2'].values,
                        z=ld_df['LD 3'].values, color=df['label'].values)

    fig.write_html(f'figures/lda/LDA3D_frac={frac:.2f}.html', auto_open=True)

# draws a 2D or 3D representation of Principal Component Analysis applied to our dataset
def visualize_PCA(df: pd.DataFrame, nr_dim: int = 2, frac: float = 1.0, standardize: bool = True):

    if nr_dim != 2 and nr_dim != 3:
        print("choose either nr_dim as 2 or 3")
        return

    df = df.sample(frac=frac)
    x, y = feature_split(df, standardize=standardize)

    # visualize differently for 2D / 3D
    if nr_dim == 2:
        y = [{'awake': 0, 'light': 1, 'deep': 2, 'rem': 3}[key] for key in y]

        pca = PCA(n_components=2)
        principal_comps = pca.fit_transform(x)

        pca_df = pd.DataFrame(data=principal_comps, columns=['PC 1', 'PC 2'])
        colors = ['green', 'blue', 'purple', 'red']

        fig = plt.figure()
        ax = plt.axes()
        ax.scatter(pca_df['PC 1'].values, pca_df['PC 2'].values,
                   c=y, cmap=matplotlib.colors.ListedColormap(colors))
        plt.savefig(f'figures/pca/PCA{nr_dim}D_frac={frac:.2f}.png')

    elif nr_dim == 3:

        pca = PCA(n_components=3)
        principal_comps = pca.fit_transform(x)

        pca_df = pd.DataFrame(data=principal_comps, columns=['PC 1', 'PC 2', 'PC 3'])

        # check distribution of number of labels
        print(f"Awake: n={len(df[df['label'] == 'awake'])}\n"
              f"Light: n={len(df[df['label'] == 'light'])}\n"
              f"Deep: n={len(df[df['label'] == 'deep'])}\n"
              f"REM: n={len(df[df['label'] == 'rem'])}\n")

        fig = px.scatter_3d(principal_comps, x=pca_df['PC 1'].values, y=pca_df['PC 2'].values,
                            z=pca_df['PC 3'].values, color=df['label'].values)

        fig.write_html(f'figures/pca/PCA{nr_dim}D_frac={frac}.html', auto_open=True)
