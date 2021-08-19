import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d


def visualize_PCA(df: pd.DataFrame, nr_dim: int = 2, frac: float = 1.0):
    # print(df)
    df = df.sample(frac=frac)

    if nr_dim != 2 and nr_dim != 3:
        print("choose either nr_dim as 2 or 3")
        return

    features = ['rr_mean', 'rr_std', 'rs_std', 'rr_range', 'gender',
                'rr_delta_abs', 'rs_delta_abs', 'rr_disp', 'rr_trend', 'age']

    x = df.reindex(columns=features).values
    y = df.reindex(columns=['label']).values

    x = StandardScaler().fit_transform(x)

    if nr_dim == 2:

        y = [{'awake': 0, 'light': 1, 'deep': 2, 'rem': 3}[key] for key in y.flatten()]

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
        plt.savefig(f'figures/pca/PCA{nr_dim}D_frac={frac}.png')

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

