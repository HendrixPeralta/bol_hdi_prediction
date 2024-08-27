# %%
import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

# %%
geo_municipalities = gpd.read_file("data/shapefile/bolivia_adm3.shp")
# %%

fig, ax = plt.subplots(figsize=(20,20))

geo_municipalities.plot(
    column='index_sdg1',
    scheme="naturalbreaks",
    cmap="OrRd",
    edgecolor="k",
    linewidth=0.2,
    legend=True,
    ax=ax
    
)

ax.set_axis_off();
ax.set_title("title")


# %%
