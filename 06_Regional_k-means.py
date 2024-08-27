# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import libpysal
from libpysal.weights import Queen, KNN

from esda.moran import Moran

import spopt
from spopt.region import RegionKMeansHeuristic

import sklearn.cluster
from sklearn.cluster import KMeans


# %%
geo_municipalities = gpd.read_file("data/shapefile/bolivia_adm3.shp")
# %%

sdg_indexes = ['index_sdg1',
                'index_sdg2',
                'index_sdg3',
                'index_sdg4',
                'index_sdg5',
                'index_sdg6',
                'index_sdg7',
                'index_sdg8',
                'index_sdg9',
                'index_sd10',
                'index_sd11',
                'index_sd13',
                'index_sd15',
                'index_sd16',
                'index_sd17',]

# Quantiles  ======================================================================= 
fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(30,40))
axs = axs.flatten()

for i, index in enumerate(sdg_indexes): 
    ax = axs[i]
    geo_municipalities.plot(
        column=index,
        scheme="Quantiles",
        cmap="OrRd",
        edgecolor="k",
        linewidth=0,
        legend=True,
        ax=ax,
        legend_kwds={
            "fontsize":17,
            "markerscale":1.5,
        }
    )

    ax.set_axis_off();
    ax.set_title(index)

plt.subplots_adjust(wspace=0.1)
plt.tight_layout()
plt.show()
# ======================================================================= Quantiles 
 

# %%
# geo_municipalities_queen_w  = libpysal.weights.fuzzy_contiguity(geo_municipalities)

# Queen Weight Matrix =======================================================================  

geo_municipalities_queen_w = Queen.from_dataframe(geo_municipalities)

f, axs = plt.subplots(figsize=(15, 15))

geo_municipalities.plot(
        edgecolor="k", facecolor="w", ax=axs
    )

geo_municipalities_queen_w.plot(
        geo_municipalities,
        ax=axs,
        edge_kws=dict(color="r", linestyle=":", linewidth=1),
        node_kws=dict(marker=""),
    )

axs.set_axis_off()
# %%
print(geo_municipalities_queen_w.n)
print(geo_municipalities_queen_w.pct_nonzero)

s = pd.Series(geo_municipalities_queen_w.cardinalities)
s.plot.hist(bins=s.unique().shape[0])


# =======================================================================  Queen Weight Matrix


# %%
np.random.seed(42)
# Calculate moran's i  - spatial autocorrelation 

# Moran's I =======================================================================  
morans_i_result = [
    Moran(geo_municipalities[index], geo_municipalities_queen_w) for index in sdg_indexes
]

# sctructure results as a list of tuples
morans_i_result=[
    (index, res.I, res.p_sim)
    for index, res in zip(sdg_indexes, morans_i_result)
]

#display table
morans_table = pd.DataFrame(
    morans_i_result,
    columns=["Index", "Moran's I", "P-value"]
).set_index("Index")

# ======================================================================= Moran's I  

#%%
# Bivariate correlation 

_ = sns.pairplot(
    geo_municipalities[sdg_indexes], kind="reg", diag_kind="kde"
)


# %%
kmeans = KMeans(n_clusters=5)
np.random.seed(42)
# model = RegionKMeansHeuristic(geo_municipalities['index_sdg1'].values, 5, geo_municipalities_queen_w)
# model.solve()
k5cls = kmeans.fit(geo_municipalities[['asdf_id', 'index_sdg1']])
k5cls.labels_[:5]



# %%
geo_municipalities["k5cls"] = k5cls
geo_municipalities.plot(
    column="k5cls",
    categorical=True,
    cmap="tab20",
    figsize=(8,8),
    edgecolor="w",
    legend=True
)
# %%
