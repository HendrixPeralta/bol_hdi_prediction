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
# geo_municipalities_w  = libpysal.weights.fuzzy_contiguity(geo_municipalities)
geo_municipalities_w = Queen.from_dataframe(geo_municipalities)

# %%
np.random.seed(42)
# Calculate moran's i 
morans_i_result = [
    Moran(geo_municipalities[index], geo_municipalities_w) for index in sdg_indexes
]

# sctructure results as a list of tuples
morans_i_result=[
    (index, res.I, res.p_sim)
    for index, res in zip(sdg_indexes, morans_i_result)
]

#display table
# morans_table = pd.DataFrame(
#     morans_i_result,
#     columns=["Index", "Moran's I", "P-value"]
# ).set_index("Index")



#%%
kmeans = KMeans(n_clusters=5)
np.random.seed(42)
# model = RegionKMeansHeuristic(geo_municipalities['index_sdg1'].values, 5, geo_municipalities_w)
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
