# %%
# %load_ext autoreload
# %autoreload 2

import re
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import xarray as xr
from geocube.api.core import make_geocube
from IPython.display import display
from pytask import Product, mark, task
from rasterio.features import rasterize
from tqdm import tqdm

from gch4i.config import global_data_dir_path, proxy_data_dir_path, sector_data_dir_path
from gch4i.gridding import GEPA_spatial_profile

pd.set_option("future.no_silent_downcasting", True)
pd.set_option("float_format", "{:f}".format)
# %%


# define a function to normalize the data by the grouping variable(s)
def normalize(x):
    return x / x.sum() if x.sum() > 0 else 0


# %%
# define the mapping between the proxy names and the USDA NASS layer names
proxy_usda_dict = {
    "broiler": "luus_lvbrltrk.shp",
    # "chicken": all of layers, pullets, briolers, roosters
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

usda_layer_name_list = list(set(proxy_usda_dict.values()))
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
dst_paths = [proxy_data_dir_path / f"{x}_proxy.nc" for x in proxy_name_list]


@mark.persist
@task(id="livestock_proxy")
def task_livestock_proxy() -> None:
    pass


luc_input_path: Path = livestock_dir_path / "LUC_ranked_US48_HI_AK.zip"
area_input_path: Path = global_data_dir_path / "gridded_area_001_cm2.tif"
# state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
county_path: str = global_data_dir_path / "tl_2020_us_county.zip"
output_paths: Annotated[list[Path], Product] = dst_paths

# %%
high_res_profile = GEPA_spatial_profile(0.01)

# %%
county_gdf = (
    gpd.read_file(county_path, columns=["GEOID", "NAME", "STATEFP", "geometry"])
    .rename(columns=str.lower)
    .astype({"geoid": int, "statefp": int})
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .to_crs(4326)
)
county_gdf.boundary.plot()

# %%

# get the layers in the zipfile

# # used to search the zipfile for the lower48 shapfiles
# shp_pat = re.compile(r"luus.*\.shp")

# with ZipFile(luc_input_path, "r") as zip_ref:
#     all_layers_list = [x for x in zip_ref.namelist() if shp_pat.match(x)]
# all_layers_list

# # we take the full layers list and get out only the ones we need.
# proxy_layer_list = [
#     x for x in all_layers_list if any([y in x for y in usda_layer_name_list])
# ]
# # if we didn't get all the layers we wanted, raise an error
# if not len(proxy_layer_list) == len(usda_layer_name_list):
#     raise ValueError("Not all layers are present")


# %%
# get the area array
with rasterio.open(area_input_path) as src:
    area_arr = src.read(1)
    area_arr = np.where(area_arr == src.nodata, 0, area_arr)

# create xarray dataset with the area array and county geoids
area_xr = xr.DataArray(
    area_arr,
    coords={"y": np.flip(high_res_profile.y), "x": high_res_profile.x},
    dims=["y", "x"],
).rio.write_crs(high_res_profile.profile["crs"])

cnty_grid = make_geocube(
    vector_data=county_gdf, measurements=["geoid"], like=area_xr, fill=99
)
cnty_grid
# %%
# loop over each of the layers of interest and read them in, assign the rank prob
usda_gdf_dict = dict()
for usda_name in tqdm(usda_layer_name_list, desc="reading in layers"):
    gdf = (
        gpd.read_file(
            f"{luc_input_path}!{usda_name}",
            columns=["RANK", "Stcopoly", "atlas_stco", "geometry"],
        )
        .rename(columns=str.lower)
        .astype({"stcopoly": int})
        # .to_crs(4326)
        # .sjoin(state_gdf[["geometry"]])
        .assign(
            area=lambda df: df.area,
            # as in v2, we calculate the recoded rank by replacing the dot rank with
            # the given probabilities, multiplied by area.
            recode_rank=lambda df: df["rank"].replace(recode_rank_dict),
        )
        .rename(columns={"stcopoly": "poly_id"})
        .set_index("poly_id")
    ).to_crs(4326)
    usda_gdf_dict[usda_name] = gdf


# %%

# loop over the geodataframes and rasterize the rank prob values
var_to_grid = "recode_rank"

usda_arr_dict = dict()
for name, gdf in tqdm(usda_gdf_dict.items(), desc=f"rasterizing {var_to_grid}"):
    gdf = usda_gdf_dict[usda_name]
    animal_arr = rasterize(
        shapes=[
            (shape, value) for shape, value in gdf[["geometry", var_to_grid]].values
        ],
        out_shape=high_res_profile.arr_shape,
        # fill=0.0,
        transform=high_res_profile.profile["transform"],
        dtype=np.float64,
        # merge_alg=rasterio.enums.MergeAlg.add,
    )
    # multiply the rank prob by the area to get the animal proxy values
    arr = animal_arr * area_arr
    arr_xr = xr.DataArray(
        arr,
        coords={"y": np.flip(high_res_profile.y), "x": high_res_profile.x},
        dims=["y", "x"],
    ).rio.write_crs(high_res_profile.profile["crs"])
    arr_xr

    # assign the county grid as a new variable in the rank dataset
    arr_xr["geoid"] = (cnty_grid.dims, cnty_grid["geoid"].values)
    # arr_xr["geoid"] = cnty_grid
    np.isnan(arr_xr["geoid"]).any()

    # apply the normalization function to the data
    out_ds = (
        arr_xr.groupby("geoid")
        .apply(normalize)
        .sortby(["y", "x"])
        .to_dataset(name=name)
    )
    out_ds[name].shape

    # check that the normalization worked
    all_eq_df = (
        out_ds[name]
        .groupby(["geoid"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
    )

    # NOTE: Due to floating point rouding, we need to check if the sum is close to 1,
    # not exactly 1.
    vals_are_one = np.isclose(all_eq_df["sum_check"], 1).all()
    print(f"are all county/year norm sums equal to 1? {vals_are_one}")
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")
    usda_arr_dict[name] = out_ds

# %%
# loop over the arrays, normalize the values by county, save the output
for output_path, (name, arr) in zip(output_paths, usda_arr_dict.items()):
    print(name, output_path.name)
    if name not in output_path.name:
        raise ValueError(f"name does not match output path: {name}, {output_path.name}")
    # %%


# %%
for proxy_name, usda_name in proxy_usda_dict.items():
    print(proxy_name, usda_name)
    if usda_name is None:
        continue
    print(proxy_name, usda_name)
    out_ds = usda_arr_dict[usda_name]
    gdf.plot(column="recode_rank", legend=True)
    plt.show()
