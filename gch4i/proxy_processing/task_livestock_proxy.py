# %%
# TODO: pytask this file

# %load_ext autoreload
# %autoreload 2

import multiprocessing
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
from geocube.api.core import make_geocube
from pytask import Product, mark, task
from rasterio.features import rasterize
from tqdm import tqdm

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
from gch4i.utils import GEPA_spatial_profile, warp_to_gepa_grid, normalize

# import re
# from zipfile import ZipFile
NUM_WORKERS = multiprocessing.cpu_count()

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("float_format", "{:f}".format)
# %%

# luus_crfldcrp.shp       field crops
# luus_crirrlnd.shp       irrigated land
# luus_crlndfrm.shp       land in farms
# luus_crnursry.shp       nurseries
# luus_crpastur.shp       pasture
# luus_crtotcrp.shp       total crops
# luus_crvgfrnt.shp       vegetables/fruits/nuts
# luus_ecallfrm.shp       all farms
# luus_lvanimal.shp       all animals
# luus_lvbeehny.shp       bees/honey
# luus_lvbrltrk.shp       chickens/turkeys
# luus_lvctlfed.shp       cattle on feed
# luus_lvctlinv.shp       cattle inventory
# luus_lvfish.shp         fish
# luus_lvgoat.shp         goats
# luus_lvhogpig.shp       hogs & pigs
# luus_lvhrspny.shp       horses/ponies
# luus_lvlyrplt.shp       layers & pullets
# luus_lvmlkcow.shp       milk cows
# luus_lvshplmb.shp       sheep & lambs

# %%
# define the mapping between the proxy names and the USDA NASS layer names
proxy_usda_dict = {
    "broiler": "luus_lvbrltrk.shp",
    "chickens": "luus_lvbrltrk.shp",
    "layers": "luus_lvlyrplt.shp",
    "pullets": "luus_lvlyrplt.shp",
    "turkeys": "luus_lvbrltrk.shp",
    "mules": "luus_lvanimal.shp",
    "beef": "luus_lvctlinv.shp",
    "cattle": "luus_lvctlinv.shp",
    "bison": "luus_lvctlinv.shp",
    "dairy": "luus_lvmlkcow.shp",
    "onfeed": "luus_lvctlfed.shp",
    "goats": "luus_lvgoat.shp",
    "horses": "luus_lvhrspny.shp",
    "sheep": "luus_lvshplmb.shp",
    "swine": "luus_lvhogpig.shp",
}

# get the unique list of usda data we're going to process
usda_layer_name_list = list(set(list(proxy_usda_dict.values())))
usda_layer_name_list

# usda_layer_name_list = list(set(proxy_usda_dict.values()))
proxy_name_list = list(set(proxy_usda_dict.keys()))

# the rank to dot liklihood values provided by the USDA NASS
recode_rank_dict = {
    6.5: 0.0,
    1: 0.0001,
    2: 0.0200,
    3: 0.0500,
    4: 0.1000,
    5: 0.3300,
    6: 0.4999,
}

livestock_dir_path = sector_data_dir_path / "USDA_NASS"
# the paths to the output files

dst_paths = [proxy_data_dir_path / f"livestock_{x}_proxy.nc" for x in proxy_name_list]


@mark.persist
@task(id="livestock_proxy")
def task_livestock_proxy() -> None:
    pass


# USDA data that holds the animal rank shapefiles
luc_input_path: Path = livestock_dir_path / "LUC_ranked_US48_HI_AK.zip"
# the high res gridded path we'll use to first calculate the rank probability
high_res_area_input_path: Path = global_data_dir_path / "gridded_area_001_cm2.tif"
# the low res area path used as a referenc to build xarray datasett
area_input_path: Path = global_data_dir_path / "gridded_area_01_cm2.tif"
# the county shapefile which is used to normalize data to the right geographic level
county_path: str = global_data_dir_path / "tl_2020_us_county.zip"
# the outputs paths for all the proxies
output_paths: Annotated[list[Path], Product] = dst_paths

# %%
high_res_profile = GEPA_spatial_profile(0.01)
out_prop = high_res_profile.profile.copy()
out_prop.update(count=1)

# create xarray dataset with the area array and county geoids
area_xr = rioxarray.open_rasterio(area_input_path)
area_xr
# %%

# get the high res area array
with rasterio.open(high_res_area_input_path) as src:
    area_profile = src.profile
    area_arr = src.read(1)
    area_arr = np.where(area_arr == src.nodata, 0, area_arr)

county_gdf = (
    gpd.read_file(county_path, columns=["GEOID", "NAME", "STATEFP", "geometry"])
    .rename(columns=str.lower)
    .astype({"geoid": int, "statefp": int})
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .to_crs(4326)
)

cnty_grid = make_geocube(
    vector_data=county_gdf, measurements=["geoid"], like=area_xr, fill=99
)
cnty_grid

# %%
var_to_grid = "recode_rank"

# %%
paths_to_warp = []
for usda_name in tqdm(usda_layer_name_list, desc="reading in layers"):

    intermediate_path_1 = livestock_dir_path / f"{usda_name.split('.')[0]}.tif"
    paths_to_warp.append(intermediate_path_1)
    # if not intermediate_path_1.exists():
    # read in the area rank vector data
    gdf = (
        gpd.read_file(
            f"{luc_input_path}!{usda_name}",
            columns=["RANK", "Stcopoly", "atlas_stco", "geometry"],
        )
        .rename(columns=str.lower)
        .astype({"stcopoly": int})
        .assign(
            area=lambda df: df.area,
            # as in v2, we calculate the recoded rank by replacing the dot rank with
            # the given probabilities.
            recode_rank=lambda df: df["rank"].replace(recode_rank_dict),
        )
        .rename(columns={"stcopoly": "poly_id"})
        .set_index("poly_id")
    ).to_crs(4326)

    # turn the vector data into rasters
    animal_arr = rasterize(
        shapes=[
            (shape, value) for shape, value in gdf[["geometry", var_to_grid]].values
        ],
        out_shape=high_res_profile.arr_shape,
        transform=high_res_profile.profile["transform"],
        dtype=np.float64,
    )
    # multiply the rank prob by the area to get the animal proxy values
    arr = animal_arr * area_arr
    # save these files to disk
    with rasterio.open(intermediate_path_1, "w", **out_prop) as dst:
        dst.write(arr, 1)


# %%

# for each of the high res raster data, warp it to the gepa grid
gepa_paths = []
for warp_path in tqdm(paths_to_warp, desc="warping to gepa grid"):
    out_path = warp_path.with_name(warp_path.stem + "_gepa.tif")
    gepa_paths.append(out_path)
    warp_to_gepa_grid(
        input_path=warp_path, output_path=out_path, num_threads=NUM_WORKERS
    )


# %%
# for each of the raster files, normalize the data by the county level, expand the dims
# to include the years, and save the data to disk by proxy mapping
for gepa_path in tqdm(gepa_paths, desc="normalizing and saving proxy"):
    gepa_path
    arr_xr = rioxarray.open_rasterio(gepa_path).squeeze("band").drop_vars("band")
    arr_xr["geoid"] = (cnty_grid.dims, cnty_grid["geoid"].values)
    out_ds = (
        arr_xr.groupby("geoid")
        .apply(normalize)
        .sortby(["y", "x"])
        .to_dataset(name="rel_emi")
        .expand_dims(year=years, axis=0)
    )
    out_ds["rel_emi"].shape
    out_ds["rel_emi"].sel(year=2020).plot()

    all_eq_df = (
        out_ds["rel_emi"]
        .groupby(["geoid"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
    )
    vals_are_one = np.isclose(all_eq_df["sum_check"], 1).all()
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")

    # print(f"are all county/year norm sums equal to 1? {vals_are_one}")
    for proxy_name, usda_name in proxy_usda_dict.items():
        if usda_name.split(".")[0] in gepa_path.stem:
            out_path = proxy_data_dir_path / f"livestock_{proxy_name}_proxy.nc"
            out_ds.rio.write_crs(high_res_profile.profile["crs"]).to_netcdf(out_path)

# %%
# county_gdf[~county_gdf.geoid.isin(np.unique(out_ds["geoid"]))]
