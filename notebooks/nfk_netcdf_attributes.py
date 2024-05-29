"""
look at the v2 list of netCDF attributes.
See which ones we need to include in v3
"""
# %%
import xarray as xr
import rioxarray
from pathlib import Path

input_path = Path(
    (
        "~/Environmental Protection Agency (EPA)/"
        "Gridded CH4 Inventory - Task 2/ghgi_v3_working/GEPA_Source_Code/"
        "Final_Gridded_Data/EPA_v2_2B8_2C2_Industry.nc"
    )
).expanduser()

xds = rioxarray.open_rasterio(input_path)
xds.attrs
# %%
