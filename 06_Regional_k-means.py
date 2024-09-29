# %%
import esda.moran
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches

import libpysal
from libpysal.weights import Queen, KNN

from esda.moran import Moran
from pysal.explore import esda
from pysal.lib import weights
from splot import esda as esdaplot

import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import robust_scale # standarizes the variables 
# db_scaled = robust_scale(db[cluster_variables]) # standarizes the variables 
# %%
geo_municipalities = gpd.read_file("data/shapefile/bolivia_adm3.shp")
# %%

sdg_indexes = [
    'index_sdg1',
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

sdg_names = [
    "SDG1: No Poverty",
    "SDG2: Zero Hunger",
    "SDG3: Good Health and Well-being",
    "SDG4: Quality Education",
    "SDG5: Gender Equality",
    "SDG6: Clean Water and Sanitation",
    "SDG7: Affordable and Clean Energy",
    "SDG8: Decent Work and Economic Growth",
    "SDG9: Industry, Innovation, and Infrastructure",
    "SDG10: Reduced Inequalities",
    "SDG11: Sustainable Cities and Communities",
    "SDG13: Climate Action",
    "SDG15: Life on Land",
    "SDG16: Peace, Justice, and Strong Institutions",
    "SDG17: Partnerships for the Goals"
]

# Quantiles  ======================================================================= 
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(60,50))

axs = axs.flatten()
# fig.subplots_adjust(top=0.9)

for i, (index, index_name) in enumerate(zip(sdg_indexes, sdg_names)): 
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
            "fontsize":30,
            "markerscale":2.5,
        }
    )

    # ax.legend(title="Natural brakes")
    
    
    ax.set_axis_off();
    ax.set_title("\n".join([index_name]), fontsize=40)

fig.suptitle("Sustainable Development Goals Geospatial Patterns In Bolivian Municipalities",
            # y=0.98,
            fontsize=75,
            fontdict={"fontweight":"bold"})

plt.subplots_adjust(wspace=0.1)
plt.tight_layout()
plt.show()
# ======================================================================= Quantiles 
 

# %%
# geo_municipalities_queen_w  = libpysal.weights.fuzzy_contiguity(geo_municipalities)

# Queen Weight Matrix =======================================================================  

geo_municipalities_queen_w = Queen.from_dataframe(geo_municipalities)
geo_municipalities_queen_w.to_file("exports/06_exports/geo_municipalities_queen_w.gal")
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
# pairplot ======================================================================= 
# _ = sns.pairplot(
#     geo_municipalities[sdg_indexes], kind="reg", diag_kind="kde"
# )
# ======================================================================= pairplot

# %%
# Multivariate K-means clustering ======================================================================= 
# TODO: kmeans

kmeans = KMeans(n_clusters=5)

np.random.seed(42)
# model = RegionKMeansHeuristic(geo_municipalities['index_sdg1'].values, 5, geo_municipalities_queen_w)
# model.solve()
k5cls = kmeans.fit(geo_municipalities[sdg_indexes])

# %%
k5cls.labels_[:5]

 # %%

geo_municipalities["k5cls"] = k5cls.labels_

fig, ax = plt.subplots(1, figsize=(12,12))

geo_municipalities.plot(
    ax=ax,
    column="k5cls",
    categorical=True,
    cmap="tab20",
    figsize=(8,8),
    edgecolor="w",
    legend=True,
    linewidth=0.2
)

ax.set_axis_off()
plt.show()

# %%
# Count the obrservations in each cluster
k5sizes = geo_municipalities.groupby("k5cls").size()

# Group clusters by label and obtain their mean 
k5means = geo_municipalities.groupby("k5cls")[sdg_indexes].mean()
k5means.T.round(3)

# Group clusters by label and describe
k5desc = geo_municipalities.groupby("k5cls")[sdg_indexes].describe()
for cluster in k5desc.T:
    print(f"{cluster}")
    print(k5desc.T[cluster].unstack())

# %%
# Draw distribution of the cluster members 
tidy_geo_municipalities = geo_municipalities[sdg_indexes + ["k5cls"]].set_index("k5cls")
# Creates a long version of the dataset
tidy_geo_municipalities = tidy_geo_municipalities.stack()
tidy_geo_municipalities = tidy_geo_municipalities.reset_index()
# Rename Columns 
tidy_geo_municipalities = tidy_geo_municipalities.rename(
    columns={"level_1":"Attribute", 0:"Values"}
)

# %% 

# Example categories (replace with your actual categories)
categories = [0, 1, 2, 3, 4]

# Generate a color palette (tab20 has 20 colors)
palette = sns.color_palette("tab20", len(categories))

# Create a color mapping dictionary
color_mapping = {category: color for category, color in zip(categories, palette)}

# %%
sns.set(font_scale=1.5)

facets = sns.FacetGrid(
    data=tidy_geo_municipalities,
    col="Attribute",
    hue="k5cls",
    sharey=False,
    sharex=False, 
    aspect=2,
    col_wrap=3,
    palette=color_mapping
)

_= facets.map(sns.kdeplot, "Values", fill=True).add_legend(title="Clusters")
# ======================================================================= Multivariate K-means clustering

# %%
# Spatial Restrained Multivariate Herargical clustering ================================================================= 

# TODO: Agglomerative clustering

np.random.seed(42)
sr_kmeans= AgglomerativeClustering(
    connectivity=geo_municipalities_queen_w.sparse, 
    n_clusters=5
)
sr_kmeans.fit(geo_municipalities[sdg_indexes])

geo_municipalities["ward5wq"] = sr_kmeans.labels_

fig, ax = plt.subplots(1, figsize=(12,12))

geo_municipalities.plot(
    column="ward5wq",
    cmap="tab20",
    categorical=True,
    legend=True,
    linewidth=0.1,
    ax=ax
)


for category, color in color_mapping.items():
    geo_municipalities[geo_municipalities["ward5wq"] == category].plot(
        color=color,
        ax=ax,
        # alpha=0.6
    )


ax.set_axis_off()

legend_patches = [
    mpatches.Patch(color=color_mapping[category], label=f"Cluster {category}")
    for category in categories
]
ax.legend(handles=legend_patches, title="Clusters", loc='upper right')
plt.show()

# %%

tidy_rs_municipalities = geo_municipalities[sdg_indexes + ["ward5wq"]].set_index("ward5wq")
# Creates a long version of the dataset
tidy_rs_municipalities = tidy_rs_municipalities.stack()
tidy_rs_municipalities = tidy_rs_municipalities.reset_index()
# Rename Columns 
tidy_rs_municipalities = tidy_rs_municipalities.rename(
    columns={"level_1":"Attribute", 0:"Values"}
)

# %% 

sns.set(font_scale=1.5)

facets = sns.FacetGrid(
    data=tidy_rs_municipalities,
    col="Attribute",
    hue="ward5wq",
    sharey=False,
    sharex=False, 
    aspect=2,
    col_wrap=3,
    palette="tab20"
)

_= facets.map(sns.kdeplot, "Values", fill=True).add_legend()

# =================================================================  Spatial Restrained Multivariate K-means clustering


# %%
# ESDA ======================================================================= 

fig, ax = plt.subplots(1, figsize=(12,12))

geo_municipalities.plot(
    column="imds",
    cmap="viridis",
    scheme="quantiles",
    k=5,
    edgecolor="white",
    linewidth=0.1,
    alpha=0.75,
    legend=True,
    legend_kwds=dict(loc=1),
    ax=ax
)

ax.set_axis_off()

# %%
lisa = esda.moran.Moran_Local(geo_municipalities["imds"], geo_municipalities_queen_w)


fig, axs = plt.subplots(2,2, figsize=(12,12))
axs = axs.flatten()

ax = axs[0]

geo_municipalities["Is" ] = lisa.Is

geo_municipalities.plot(
    column="Is",
    cmap="plasma",
    scheme="quantiles",
    k=5,
    edgecolor="white",
    linewidth=0.1,
    alpha=0.75,
    legend=True,
    ax=ax 
)
ax = axs[1]
esdaplot.lisa_cluster(lisa, geo_municipalities, p=1, ax=ax)

ax = axs[2]

labels = pd.Series(
    1*(lisa.p_sim <0.05),
    index=geo_municipalities.index
).map({1:"Significant", 0:"Non-significant"})

geo_municipalities["cl"] = labels
geo_municipalities.plot(
    column="cl",
    categorical=True,
    k=2,
    cmap="Paired",
    linewidth=0.1,
    edgecolor = "white",
    legend=True,
    ax=ax
)

ax = axs[3]
esdaplot.lisa_cluster(lisa, geo_municipalities, p=0.05, ax=ax)

for i, ax in enumerate(axs):
    ax.set_axis_off()
    ax.set_title(
        ["Local Statistics",
        "Scatterplot Quadrant",
        "Statistical Significance",
        "Moran Cluster Map"][i]
    )

fig.tight_layout()
plt.show()


# ======================================================================= ESDA

# %%

# k5cls = kmeans.fit(geo_municipalities[sdg_indexes])
# k5cls.labels_[:5]
# %%


 # %%

# geo_municipalities["k5cls"] = k5cls.labels_

# fig, ax = plt.subplots(1, figsize=(12,12))

# geo_municipalities.plot(
#     ax=ax,
#     column="k5cls",
#     categorical=True,
#     cmap="tab20",
#     figsize=(8,8),
#     edgecolor="w",
#     legend=True
# )

# ax.set_axis_off()
# plt.show()

# %%

# fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(30,40))
# axs = axs.flatten()

# for i, index in enumerate(sdg_indexes): 
#     ax = axs[i]
#     geo_municipalities.plot(
#         column=index,
#         scheme="Quantiles",
#         cmap="OrRd",
#         edgecolor="k",
#         linewidth=0,
#         legend=True,
#         ax=ax,
#         legend_kwds={
#             "fontsize":17,
#             "markerscale":1.5,
#         }
#     )

#     ax.set_axis_off();
#     ax.set_title(index)

# plt.subplots_adjust(wspace=0.1)
# plt.tight_layout()
# plt.show()


#%%

