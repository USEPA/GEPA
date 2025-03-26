# %%
from pathlib import Path
import numpy as np
import rioxarray
import xarray as xr

from gch4i.config import tmp_data_dir_path
from gch4i.utils import write_tif_output

v2_in_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/Gridded_GHGI_Methane_v2_2018.nc"
)
v3_in_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/gridding_log_and_qc/gridded_output/1B1a_abandoned_coal-abd_coal_emi-abd_coal_proxy.tif"
)

pop_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/sector/worldpop/usa_ppp_2012_reprojected.tif"
)

v2_ds = xr.open_dataset(v2_in_path)
v3_ds = xr.open_dataset(v3_in_path)
pop_ds = xr.open_dataset(pop_path)
# %%
lon_left = -129.95  # deg
lon_right = -59.95  # deg
lat_up = 54.95  # deg
lat_low = 20.05  # deg
resolution = 0.1

x = np.linspace(lon_left, lon_right, 700)
y = np.linspace(lat_up, lat_low, 350)
# %%
x = np.arange(lon_left, lon_right, resolution)
y = np.arange(lat_up, lat_low, -resolution)
print(len(x), len(y))
# %%


v3_aligned = v3_ds.assign_coords(x=x, y=y)
v3_aligned = v3_aligned.rename({"x": "lon", "y": "lat"})
v3_aligned = v3_aligned.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
v3_aligned = v3_aligned.rio.write_crs("EPSG:4326")
# v3_alinged_ds = v3_alinged_ds.rio.write_crs("EPSG:4326", inplace=True)
# v3_alinged_ds.rio.write_nodata(0, inplace=True)
# v3_alinged_ds.rio.set_transform("EPSG:4326", inplace=True)
v3_aligned.to_netcdf(tmp_data_dir_path / "v3_aligned.nc", mode="w", format="NETCDF4")
# %%
max_v2_lat = np.max(v2_ds.lat.values)
min_v2_lat = np.min(v2_ds.lat.values)
max_v2_lon = np.max(v2_ds.lon.values)
min_v2_lon = np.min(v2_ds.lon.values)
# %%
v2_x = np.linspace(min_v2_lon, max_v2_lon, 350)
v2_y = np.linspace(min_v2_lat, max_v2_lat, 700)
print(len(v2_x), len(v2_y))
# %%
v2_x
# %%
v2_lat_vals = v2_ds.lat.values
v2_lat_vals
# %%
v2_lon_vals = v2_ds.lon.values
v2_lon_vals
# %%
np.min(v2_lon_vals)
# %%
v2_ds.lat.max()
# %%
from pathlib import Path
import xarray as xr
from gch4i.config import global_data_dir_path, proxy_data_dir_path

area_path = Path(global_data_dir_path / f"gridded_area_01_cm2.tif")

area_ds = xr.open_dataset(area_path)

gridded_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/proxy"
)


area_y_flipped = area_ds.y.values[::-1]

for in_path in gridded_path.glob("*.nc"):
    tmp_ds = xr.open_dataset(in_path)
    x_matches = all(tmp_ds.x.values == area_ds.x.values)
    y_matches = all(tmp_ds.y.values == area_y_flipped)
    if any([not x_matches, not y_matches]):
        print(f"WARNING: x or y values do not match for {in_path.name}")
        print(f"    x matches: {x_matches}")
        print(f"    y matches: {y_matches}")

# %%
import rioxarray
rioxarray.open_rasterio(in_path)
# %%
another_path = Path(proxy_data_dir_path / "population_proxy.nc")
tmp_ds = xr.open_dataset(another_path)
# %%
x_matches = all(out_ds.x.values == area_ds.x.values)
y_matches = all(out_ds.y.values == area_ds.y.values)
all([x_matches, y_matches])
# %%
out_ds.x.min()

# %%
area_ds.x.min()
# %%
