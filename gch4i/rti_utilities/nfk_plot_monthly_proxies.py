# %%
# Import necessary libraries
from gch4i.config import logging_dir, global_data_dir_path, prelim_gridded_dir
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import geopandas as gpd
from rasterio.plot import show
# %%
monthly_proxy_paths = list(prelim_gridded_dir.rglob("*monthly_scaling.tif"))
# monthly_proxy_paths = list(logging_dir.rglob("*proxy_monthly.tif"))

print(len(monthly_proxy_paths), "monthly proxy files found.")

# %%
state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
state_gdf = (
    gpd.read_file(state_path)
    .rename(columns=str.lower)
    .astype({"statefp": int})
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .loc[:, ["geometry"]]
    .to_crs(4326)
    .dissolve()
    .simplify(tolerance=0.5, preserve_topology=True)
)
state_gdf.boundary.plot()
# %%

ncol = 10
nrow = len(monthly_proxy_paths) // ncol + (len(monthly_proxy_paths) % ncol > 0)

fig, axs = plt.subplots(nrow, ncol, figsize=(15, 15), sharex=True)

for ax, in_path in zip(axs.ravel(), monthly_proxy_paths):
    # Read the raster data
    with rasterio.open(in_path) as src:
        data = src.read(1)
        raster_src = src.profile

    data = np.where(data == 0, np.nan, data)  # Replace negative values with NaN
    show(data, ax=ax, cmap="Set3", transform=raster_src["transform"])
    state_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)
    ax.set(title="_".join(in_path.stem.split("_")[-3:-1]))

    ax.set_axis_off()
plt.show()

# %%


