"""
Name:                   task_fbar_proxy.py
Date Last Modified:     2025-02-19
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
import numpy as np

import geopandas as gpd

from shapely.geometry import Point

from gch4i.config import (
    sector_data_dir_path,
    proxy_data_dir_path,
    global_data_dir_path,
    emi_data_dir_path,
    years
)

########################################################################################
# %% STEP 0.2. Load Path Files

# state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"

# GEPA_Field_Burning_Path = (
#     V3_DATA_PATH.parent / "GEPA_Source_Code" / "GEPA_Field_Burning"
# )

# InputData_path = GEPA_Field_Burning_Path / "InputData/"

McCarty_years = np.arange(2003, 2008)

########################################################################################
# %% Proxy Functions


def create_circle_geometry(row, radius_crs="EPSG:3857"):
    """ "
    Create a circular polygon centered on a lat/lon point using a radius derived from
    the "Acres" columns.
    """
    # Convert Acres to radius in meters
    area_m2 = row["Acres"] * 4046.86  # Convert acres to m^2
    radius_m = np.sqrt(area_m2 / np.pi)

    # Create a point geometry
    center = Point(row["geometry"])

    # Project teh cente rpoint to a metric CRS (EPSG:3857)
    gdf_point = gpd.GeoDataFrame({"geometry": [center]}, crs="EPSG:4326")
    gdf_point = gdf_point.to_crs(radius_crs)

    # Buffer in metric CRS
    buffered = gdf_point.geometry.buffer(radius_m)

    # Reproject back to EPSG:4326
    buffered = gpd.GeoDataFrame({"geometry": buffered}, crs=radius_crs).to_crs(
        "EPSG:4326"
    )

    # Return the geometry
    return buffered.iloc[0].geometry


# Base proxy retrieval function. To be used with 7 pytask functions
def get_fbar_proxy_data(
    filter_condition, state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
):
    """
    This base proxy function will be used in concert with 7 pytask functions to retrieve
    relevant state emissions data, fill in gaps in proxy data from McCarty, et al.
    (2011), and calculate relative emissions per crop proxy.

    Step 1: Build the base proxy data from McCarty, et al. (2011) data
    Step 2: Check state emissions for emi data without proxy data
    Step 3: Transform Lat/Lon and Acres to Geometry Polygons
    Step 4: Append missing Maine (ME) Proxy data, if necessary
    Step 5: Calculate rel_emi (month level)
    Step 6: Explode Years
    """

    # STEP 1. Build base_proxy
    base_proxy = pd.DataFrame()

    # Map month to month number
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    # 14 blank burn dates in 2003 data, removed
    for year in McCarty_years:
        df = (
            # Read in McCarty, et al. (2011) data
            pd.read_csv(
                f"{sector_data_dir_path}/fbar/{year}_ResidueBurning_AllCrops.csv",
                usecols=[
                    "Crop_Type",
                    "Acres",
                    "EMCH4_Gg",
                    "Burn_Date",
                    "Latitude",
                    "Longitude",
                ],
            )
            # Drop rows with missing burn date values
            .dropna(subset=["Burn_Date"])
            .query("Burn_Date != ' '")
        )
        df = (
            # Clean and assign month and crop
            df.assign(
                month=lambda x: x["Burn_Date"].str.extract(r"([A-Za-z]{3})")[0]
                .map(month_map)
                .astype(int),
                crop=lambda x: np.select(
                    condlist=[
                        x["Crop_Type"] == "corn",
                        x["Crop_Type"].isin(
                            ["other crop/fallow", "Kentucky bluegrass", "lentils"]
                        ),
                    ],
                    choicelist=["maize", "other"],
                    default=x["Crop_Type"],
                )
            )
            # Remove unnecessary columns
            .drop(columns=["Burn_Date", "Crop_Type"])
        )

        # Append to all_data DataFrame
        base_proxy = pd.concat([base_proxy, df], ignore_index=True)

    # Convert lat/lon to geometry points
    base_proxy = gpd.GeoDataFrame(
        base_proxy,
        geometry=gpd.points_from_xy(base_proxy.Longitude, base_proxy.Latitude),
        crs="EPSG:4326",
    ).drop(columns=["Latitude", "Longitude"])

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

    # Assign Each row to a State
    base_proxy = (
        gpd.sjoin(base_proxy, state_gdf, how="left", predicate="within")
        .drop(columns=["index_right", "state_name", "statefp"])
        .loc[:, ["month", "crop", "state_code", "Acres", "EMCH4_Gg", "geometry"]]
    )

    ################################################################################
    # STEP 2. Check crop for emi-proxy match

    # Build dictionary to map crop to crop emissions data
    proxy_dict = {
        "maize": "maize_emi",
        "cotton": "cotton_emi",
        "rice": "rice_emi",
        "soybean": "soybeans_emi",
        "sugarcane": "sugarcane_emi",
        "wheat": "wheat_emi",
        "other": [
            "barley_emi",
            "chickpeas_emi",
            "drybeans_emi",
            "lentils_emi",
            "oats_emi",
            "other_grains_emi",
            "peanuts_emi",
            "peas_emi",
            "potatoes_emi",
            "sorghum_emi",
            "sugarbeets_emi",
            "sunflower_emi",
            "tobacco_emi",
            "vegetables_emi",
        ],
    }

    # Filter base_proxy for current_proxy
    filtered_proxy = base_proxy.query("crop == @filter_condition")
    # Create filtered dictionary
    filtered_dict = {filter_condition: proxy_dict[filter_condition]}

    # Check if proxy data exists for emissions data
    for key, value in filtered_dict.items():
        # If value is a list, concatenate all dataframes
        if isinstance(value, list):
            emi_df = pd.concat(
                [pd.read_csv(f"{emi_data_dir_path}/{val}.csv") for val in value],
                ignore_index=True,
            ).query("ghgi_ch4_kt != 0")
        # If value is a string, read in the single file
        else:
            emi_df = pd.read_csv(f"{emi_data_dir_path}/{value}.csv").query(
                "ghgi_ch4_kt != 0"
            )

    # Retrieve unique state codes for emissions without proxy data
    # This step is necessary, as not all emissions data excludes emission-less states
    emi_states = set(emi_df["state_code"].unique())
    proxy_states = set(filtered_proxy["state_code"].unique())

    missing_states = emi_states.difference(proxy_states)

    ################################################################################
    # STEP 2A/2B. Append proxy_gdf with generalized proxy for missing states
    proxy_gdf = (
        base_proxy
        # Sum emissions
        .groupby(["month", "crop", "state_code", "Acres", "geometry"], as_index=False)
        .agg({"EMCH4_Gg": "sum"})
        # Calculate state annual relative emissions (emissions / state emissions)
        .assign(
            state_crop_sum=lambda x: x.groupby(["state_code", "crop"])[
                "EMCH4_Gg"
            ].transform("sum"),
            annual_rel_emi=lambda x: x["EMCH4_Gg"] / x["state_crop_sum"],
        )
        # Filter for current crop
        .query("crop == @filter_condition")
        .drop(columns=["state_crop_sum", "crop"])
        # Set geometry and CRS
        .set_geometry("geometry")
        .set_crs("EPSG:4326")
    )

    # Check if missing_states is empty
    if missing_states:
        alt_proxy = (
            base_proxy
            # Create generalized state relative emissions (without respect to crop)
            .assign(
                state_sum=lambda x: x.groupby(["state_code"])["EMCH4_Gg"].transform(
                    "sum"
                ),
                annual_rel_emi=lambda x: x["EMCH4_Gg"] / x["state_sum"],
            )
            .drop(columns=["state_sum", "crop"])
            # Filter for missing states only
            .query("state_code in @missing_states")
            # Set geometry and CRS
            .set_geometry("geometry")
            .set_crs("EPSG:4326")
        )
        # Append to proxy_gdf
        proxy_gdf = pd.concat([proxy_gdf, alt_proxy], ignore_index=True)

    ################################################################################
    # STEP 3. Transform Lat/Lon and Acres to Geometry Polygons
    proxy_gdf = (
        proxy_gdf.assign(geometry=lambda x: x.apply(create_circle_geometry, axis=1))
        .drop(columns=["Acres"])
        .loc[:, ["state_code", "month", "EMCH4_Gg", "annual_rel_emi", "geometry"]]
    )

    ################################################################################
    # Step 4. Append missing Maine (ME) Proxy data, if necessary

    # state: ME
    # month: Jan-Dec
    # annual_rel_emi: Proportional to generic
    # geometry: Polygon of ME

    # Maine (ME) is missing proxy data for 'other' crop
    if filter_condition == "other":
        # Create Maine (ME) geometry
        ME_geom = state_gdf.query("state_code == 'ME'")["geometry"].values[0]

        # Get the Proportional EMCH4_Gg per month
        ME_gdf = (
            base_proxy.groupby("month", as_index=False)
            .agg({"EMCH4_Gg": "sum"})
            .assign(
                ch4_sum=lambda x: x["EMCH4_Gg"].sum(),
                annual_rel_emi=lambda x: x["EMCH4_Gg"] / x["ch4_sum"],
                geometry=ME_geom,
                state_code="ME",
            )
            .drop(columns=["ch4_sum"])
            .loc[:, ["state_code", "month", "EMCH4_Gg", "annual_rel_emi", "geometry"]]
        )

        # Add ME data to proxy_gdf
        proxy_gdf = pd.concat([proxy_gdf, ME_gdf], ignore_index=True)
        # Set geometry and CRS
        proxy_gdf = gpd.GeoDataFrame(proxy_gdf, geometry="geometry", crs="EPSG:4326")

    ################################################################################
    # STEP 5. Calculate rel_emi (month level)
    proxy_gdf = (
        proxy_gdf
        .assign(
            month_sum=lambda x: x.groupby(["month"])["EMCH4_Gg"].transform("sum"),
            rel_emi=lambda x: x["EMCH4_Gg"] / x["month_sum"]
        )
        .drop(columns=["EMCH4_Gg", "month_sum"])
    )

    ################################################################################
    # STEP 6. Explode Years & make year_month
    proxy_gdf = (
        proxy_gdf.assign(
            year=lambda df: [years for _ in range(df.shape[0])]
        )
        .explode("year")
        .assign(
            year_month=lambda df: df["year"].astype(str)
            + "-"
            + df["month"].astype(str)
        )
        .loc[
            :,
            ["state_code", "year_month", "year", "month", "annual_rel_emi", "rel_emi", "geometry"]]
    )

    # Return proxy data
    return proxy_gdf


########################################################################################
########################################################################################
# %% Pytask


# Cotton Proxy
@mark.persist
@task(id="fbar_proxy")
def task_get_fbar_cotton_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "fbar_cotton_proxy.parquet",
):
    proxy_gdf = get_fbar_proxy_data(filter_condition="cotton", state_path=state_path)
    proxy_gdf.to_parquet(output_path)
    return None


# Maize Proxy
@mark.persist
def task_get_fbar_maize_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "fbar_maize_proxy.parquet",
):
    proxy_gdf = get_fbar_proxy_data(filter_condition="maize", state_path=state_path)
    proxy_gdf.to_parquet(output_path)
    return None


# Rice Proxy
@mark.persist
def task_get_fbar_rice_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "fbar_rice_proxy.parquet",
):
    proxy_gdf = get_fbar_proxy_data(filter_condition="rice", state_path=state_path)
    proxy_gdf.to_parquet(output_path)
    return None


# Soybean Proxy
@mark.persist
def task_get_fbar_soybeans_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "fbar_soybeans_proxy.parquet",
):
    proxy_gdf = get_fbar_proxy_data(filter_condition="soybean", state_path=state_path)
    proxy_gdf.to_parquet(output_path)
    return None


# Sugarcane Proxy
@mark.persist
def task_get_fbar_sugarcane_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "fbar_sugarcane_proxy.parquet",
):
    proxy_gdf = get_fbar_proxy_data(filter_condition="sugarcane", state_path=state_path)
    proxy_gdf.to_parquet(output_path)
    return None


# Wheat Proxy
@mark.persist
def task_get_fbar_wheat_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "fbar_wheat_proxy.parquet",
):
    proxy_gdf = get_fbar_proxy_data(filter_condition="wheat", state_path=state_path)
    proxy_gdf.to_parquet(output_path)
    return None


# Other Proxy
@mark.persist
def task_get_fbar_other_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "fbar_other_proxy.parquet",
):
    proxy_gdf = get_fbar_proxy_data(filter_condition="other", state_path=state_path)
    proxy_gdf.to_parquet(output_path)
    return None
