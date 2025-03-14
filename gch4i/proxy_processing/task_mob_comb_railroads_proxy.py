"""
Name:                   task_mob_comb_railroads_proxy.py
Date Last Modified:     2025-01-24
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of mobile combustion railroad proxy emissions
Input Files:            - Raw Input: {global_data_dir_path}/raw/
                        - Railroad Files: {global_data_dir_path}/raw/
                            tl_{year}_us_rails.parquet
                        - State Geo: {global_data_dir_path}/tl_2020_us_state.zip
Output Files:           - {proxy_data_dir_path}/railroads_proxy.parquet
"""

########################################################################################
# %% Load Packages

from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pytask import Product, mark, task
from tqdm.auto import tqdm

from gch4i.config import global_data_dir_path, proxy_data_dir_path, years
from gch4i.utils import normalize

parallel = Parallel(n_jobs=-1, verbose=10)

########################################################################################
# %% Define variable constants
# year_range = [*range(min_year, max_year+1, 1)]

########################################################################################
# %% Functions
# @mark.persist
# @task(id="railroads_proxy")
# def task_get_railroads_proxy(
raw_path = global_data_dir_path / "raw"

rail_input_paths = []
for year in years:
    rail_input_paths.append(raw_path / f"tl_{year}_us_rails.parquet")

# %%
# ):
"""
Process railroad geometries by state. Multilinestring geometries are exploded into
individual linestrings and clipped to state boundaries.

`raw_path` is the path to raw railroad data; however, it is not included in the
pytask function arguments because it is not a file, but a directory of files

Args:
    state_path: GeoDattaFrame containing state geometries.

Returns:
    reporting_proxy_output: GeoDataFrame containing processed railroad geometries.
"""


# Raw Railroad Data Path
# %%


def state_overlay(input_path, state_in_path):

    the_year = input_path.name.split("_")[1]
    out_path = raw_path / f"railroads_{the_year}_overlay.parquet"
    if not out_path.exists():
        state_gdf = (
            gpd.read_file(state_in_path)
            .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
            .rename(columns=str.lower)
            .rename(columns={"stusps": "state_code", "name": "state_name"})
            .astype({"statefp": int})
            # get only lower 48 + DC
            .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
            .to_crs(4326)
        )
        year_rail_gdf = gpd.read_parquet(
            input_path, columns=["MTFCC", "geometry"]
        ).to_crs(4326)
        year_rail_gdf = gpd.overlay(year_rail_gdf, state_gdf, how="intersection")
        year_rail_gdf.to_parquet(out_path, index=False)
    else:
        year_rail_gdf = gpd.read_parquet(out_path)

    return year_rail_gdf


# %%


@mark.persist
@task(id="railroads_proxy")
def task_railroad_proxy(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    input_paths: list[Path] = rail_input_paths,
    output_proxy_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "railroads_proxy.parquet"
    ),
):
    # %%
    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )
    # %%
    rail_list = parallel(
        delayed(state_overlay)(input_path=x, state_in_path=state_path)
        for x in input_paths
    )
    # %%
    rail_list_w_years = []
    for year, rail_gdf in zip(years, rail_list):
        geom_mask = (~rail_gdf.is_empty) | (rail_gdf.is_valid)
        print(f"Number of invalid geometries for {year}: {(~geom_mask).sum()}")
        rail_list_w_years.append(
            rail_gdf.assign(year=year).drop(columns="MTFCC").dissolve("state_code")
        )

    # %%
    rail_proxy_gdf = pd.concat(rail_list_w_years).reset_index().assign(rel_emi=1)
    rail_proxy_gdf
    # %%

    all_eq_df = (
        rail_proxy_gdf.groupby(["state_code", "year"])["rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(
            is_close=lambda df: (np.isclose(df["sum_check"], 1, atol=0, rtol=0.00001))
        )
    )

    if all_eq_df["is_close"].all():
        rail_proxy_gdf.to_parquet(output_proxy_path)
        print("YAY")
    else:
        raise ValueError("not all year values are normed correctly!")
    # %%
