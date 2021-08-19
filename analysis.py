import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch_train import draw_conf
from mpl_toolkits import mplot3d


def feature_split(df: pd.DataFrame, standardize: bool = True) -> (np.ndarray, np.ndarray):

    features = ['rr_mean', 'rr_std', 'rs_std', 'rr_range', 'gender',
                'rr_delta_abs', 'rs_delta_abs', 'rr_disp', 'rr_trend', 'age']

    x = df.reindex(columns=features).values
    y = df.reindex(columns=['label']).values.flatten()

    if standardize:
        x = StandardScaler().fit_transform(x)

    return x, y


def visualize_LDA(df: pd.DataFrame, frac: float = 1.0, standardize: bool = True):

    df = df.sample(frac=frac)
    x, y = feature_split(df, standardize=standardize)

    lda = LinearDiscriminantAnalysis(n_components=3).fit(x, y)
    x_r_lda = lda.transform(x)

    out = lda.predict(x)
    draw_conf(out, y, name=f"LDA_conf_frac={frac:.2f}.png")

    ld_df = pd.DataFrame(data=x_r_lda, columns=['LD 1', 'LD 2', 'LD 3'])

    fig = px.scatter_3d(x=ld_df['LD 1'].values, y=ld_df['LD 2'].values,
                        z=ld_df['LD 3'].values, color=df['label'].values)
    # fig.show()
    fig.write_html(f'figures/lda/LDA3D_frac={frac:.2f}.html', auto_open=True)


def visualize_PCA(df: pd.DataFrame, nr_dim: int = 2, frac: float = 1.0, standardize: bool = True):

    if nr_dim != 2 and nr_dim != 3:
        print("choose either nr_dim as 2 or 3")
        return

    df = df.sample(frac=frac)
    x, y = feature_split(df, standardize=standardize)

    if nr_dim == 2:

        y = [{'awake': 0, 'light': 1, 'deep': 2, 'rem': 3}[key] for key in y]

        pca = PCA(n_components=2)
        principal_comps = pca.fit_transform(x)

        pca_df = pd.DataFrame(data=principal_comps, columns=['PC 1', 'PC 2'])
        colors = ['green', 'blue', 'purple', 'red']

        fig = plt.figure()
        # cb = plt.colorbar()
        # ax = plt.axes(projection='3d')
        ax = plt.axes()
        ax.scatter(pca_df['PC 1'].values, pca_df['PC 2'].values,
                   c=y, cmap=matplotlib.colors.ListedColormap(colors))
        # plt.show()
        plt.savefig(f'figures/pca/PCA{nr_dim}D_frac={frac:.2f}.png')

    elif nr_dim == 3:

        pca = PCA(n_components=3)
        principal_comps = pca.fit_transform(x)

        pca_df = pd.DataFrame(data=principal_comps, columns=['PC 1', 'PC 2', 'PC 3'])
        colors = ['green', 'blue', 'purple', 'red']
        print(f"Awake: n={len(df[df['label'] == 'awake'])}\n"
              f"Light: n={len(df[df['label'] == 'light'])}\n"
              f"Deep: n={len(df[df['label'] == 'deep'])}\n"
              f"REM: n={len(df[df['label'] == 'rem'])}\n")
        # pca_df = pca_df.loc[1:10]

        fig = px.scatter_3d(principal_comps, x=pca_df['PC 1'].values, y=pca_df['PC 2'].values,
                            z=pca_df['PC 3'].values, color=df['label'].values)
        # fig.show()
        fig.write_html(f'figures/pca/PCA{nr_dim}D_frac={frac}.html', auto_open=True)

