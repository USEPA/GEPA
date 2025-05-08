"""
Name:                   task_fbar_proxy.py
Date Last Modified:     2024-12-09
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of field burning emissions to proxy data.
Input Files:            - McCarty et al. (2009) field burning emissions data
Output Files:           - 7 fbar proxies
Notes:                  - pytask.parameterize() is outdated.
                        - Created get_proxy_function and used with 7 pytask functions
                        - Currently fails the base function, but 7 proxy outputs are
                        correct.
                        - Other Proxy is missing proxy data for ME, and there is no
                        crop data to generalize. SOLUTION: generalize the emissions
                        across the state polygon with averaged generalized proportions
                        calculated from all other monthly crop emissions.
"""

########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import geopandas as gpd
from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    emi_data_dir_path,
    years,
)
import numpy as np

from gch4i.utils import normalize

########################################################################################
# %% STEP 0.2. Load Path Files

state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"

GEPA_Field_Burning_Path = (
    V3_DATA_PATH.parent / "GEPA_Source_Code" / "GEPA_Field_Burning" / "InputData/"
)

input_data_paths = list(GEPA_Field_Burning_Path.rglob("*ResidueBurning_AllCrops.csv"))
input_data_paths = [path for path in input_data_paths if path.is_file()]
input_data_paths

# these are the columns used to read in the csv files
fb_cols = [
    "Crop_Type",
    "Acres",
    "EMCH4_Gg",
    "Burn_Date",
    "Latitude",
    "Longitude",
]

# this is the crosswalk between the proxy name and the crop_type given in the field
# burning datasets
proxy_to_fbar_dict = {
    "maize": ["corn"],
    "cotton": ["cotton"],
    "rice": ["rice"],
    "soybeans": ["soybean"],
    "sugarcane": ["sugarcane"],
    "wheat": ["wheat"],
    "other": ["other crop/fallow", "Kentucky bluegrass", "lentils"],
}

output_path_list = [
    proxy_data_dir_path / f"fbar_{x}_proxy.parquet" for x in proxy_to_fbar_dict.keys()
]


# @mark.persist
# @task(id="fbar_proxy")
# def task_field_burning_proxy(
#     state_path: Path = state_path,
#     input_data_paths: list[Path] = input_data_paths,
#     proxy_to_fbar_dict: dict = proxy_to_fbar_dict,
#     output_path_list: Annotated[list[Path], Product] = output_path_list,
# ):

# Read in State data
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

fb_df_list = [pd.read_csv(x, usecols=fb_cols) for x in input_data_paths]
raw_df = pd.concat(fb_df_list, ignore_index=True)

raw_count = raw_df.shape[0]
print(f"Number of records: {raw_count:,}")
raw_df.isna().sum()

empty_burn_date_count = raw_df[raw_df["Burn_Date"] == " "].shape[0]
print(f"Number of empty Burn_Date entries: {empty_burn_date_count}")

fb_df = (
    raw_df.rename(columns=str.lower)
    .dropna(subset=["latitude", "longitude"], how="any")
    .assign(
        month_str=lambda df: df["burn_date"].str.extract(r"([A-Za-z]{3})")[0],
    )
    .dropna(subset="month_str")
    .assign(
        month=lambda df: pd.to_datetime(df["month_str"], format="%b")
        .dt.strftime("%m")
        .astype(int),
    )
)
fb_gdf = (
    gpd.GeoDataFrame(
        fb_df,
        geometry=gpd.points_from_xy(fb_df["longitude"], fb_df["latitude"]),
        crs="EPSG:4326",
    )
    .drop(columns=["latitude", "longitude"])
    .sjoin(state_gdf[["state_code", "geometry"]], how="left")
)
pre_state_na_count = fb_gdf.shape[0]
print(f"Number of records: {pre_state_na_count:,}")
print(f"number of records dropped: {raw_count - pre_state_na_count:,}")

na_state_code_count = fb_gdf["state_code"].isna().sum()
print(f"Number of records with NA state_code: {na_state_code_count}")
ax = fb_gdf[fb_gdf["state_code"].isna()].plot(color="xkcd:scarlet")
state_gdf.boundary.plot(ax=ax, color="xkcd:slate")
fb_gdf = fb_gdf.dropna(subset=["state_code"], how="any")
print(f"number of records dropped: {pre_state_na_count - fb_gdf.shape[0]:,}")
# %%
for (proxy_name, fbar_crop_list), out_path in zip(
    proxy_to_fbar_dict.items(), output_path_list
):
    # out_path = proxy_data_dir_path / f"fbar_{proxy_name}_proxy.parquet"
    print(proxy_name, out_path.exists())

    crop_gdf = fb_gdf[fb_gdf["crop_type"].isin(fbar_crop_list)].copy()

    crop_months = crop_gdf["month"].drop_duplicates().sort_values().to_list()
    print(f"crop_months: {len(crop_months)}")
    if len(crop_months) < 12:
        print(crop_months)

    crop_states = crop_gdf["state_code"].drop_duplicates()
    missing_states = state_gdf[~state_gdf["state_code"].isin(crop_states)]

    fill_in_data = fb_gdf[fb_gdf["state_code"].isin(missing_states.state_code)]
    fill_in_data
    print(f"filling in crop data: {fill_in_data.state_code.unique()}")
    if not fill_in_data.empty:
        crop_gdf = pd.concat([crop_gdf, fill_in_data], ignore_index=True)

    crop_states = crop_gdf["state_code"].drop_duplicates()
    missing_states = state_gdf[~state_gdf["state_code"].isin(crop_states)]

    print(f"missing_states: {missing_states['state_code'].to_list()}")
    missing_states = pd.concat(
        [missing_states.assign(month=month) for month in range(1, 13)],
        ignore_index=True,
    )
    missing_states = missing_states.assign(emch4_gg=1)

    # This should now have a comprehensive list of all states and months
    # represented in the data. If not, we need to fix something
    year_gdf = pd.concat([crop_gdf, missing_states], ignore_index=True).loc[
        :, ["state_code", "geometry", "month", "emch4_gg"]
    ]
    state_check = (
        year_gdf["state_code"].drop_duplicates().isin(state_gdf["state_code"]).all()
    )
    print(f"are all states accounted for?: {state_check}")

    # we now explode out the 1 year of data to all years of our study period
    out_crop_gdf = (
        year_gdf.assign(year=lambda df: [years for _ in range(df.shape[0])])
        .explode("year")
        .assign(
            year_month=lambda df: pd.to_datetime(
                df[["year", "month"]].assign(DAY=1)
            ).dt.strftime("%Y-%m"),
        )
    )
    # we create the emissions data for the monthly scaling factor
    out_crop_gdf["annual_rel_emi"] = out_crop_gdf.groupby(["state_code", "year"])[
        "emch4_gg"
    ].transform(normalize)
    # we create the relative emissions for each month
    out_crop_gdf["rel_emi"] = out_crop_gdf.groupby(["state_code", "year_month"])[
        "emch4_gg"
    ].transform(normalize)

    ax = out_crop_gdf.query("year == 2022").plot(color="xkcd:scarlet", markersize=1)
    state_gdf.boundary.plot(color="xkcd:slate", ax=ax)
    ax.set_title(f"{proxy_name} proxy")

    all_years = out_crop_gdf.year.drop_duplicates().isin(years).all()
    print(f"are all years accounted for?: {all_years}")

    # we make sure the monthly scaling factors pass a quick check
    annual_all_eq_df = (
        out_crop_gdf.groupby(["state_code", "year"])["annual_rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(
            is_close=lambda df: (
                np.isclose(df["sum_check"], 1, atol=0, rtol=0.00001)
            )
        )
    )
    annual_all_eq_df

    # we make sure the relative emissions for year_month pass
    if not annual_all_eq_df["is_close"].all():
        raise ValueError("not all annual values are normed correctly!")
    all_eq_df = (
        out_crop_gdf.groupby(["state_code", "year_month"])["rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(
            is_close=lambda df: (
                np.isclose(df["sum_check"], 1, atol=0, rtol=0.00001)
            )
        )
    )
    all_eq_df

    if not all_eq_df["is_close"].all():
        raise ValueError("not all year_month values are normed correctly!")

    out_crop_gdf.to_parquet(out_path, index=False)


# %%
