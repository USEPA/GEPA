"""
Name:                   task_forest_land_proxy.py
Date Last Modified:     2024-06-10
Authors Name:           C. Coxen, Nick Kruskamp (RTI International)
Purpose:                Generate proxy data for forest land remaining forest land emissions
Input Files:            -   {sector_data_dir_path}/forestlands_grasslands/MTBS_byEventFuelFuelbed_09Sep2024.csv
                        -   {sector_data_dir_path}/forestlands_grasslands/fccs_fuelbed_Aug2023_jesModified.csv
                        -   {sector_data_dir_path}/forestlands_grasslands/nawfd_fuelbed_Aug2023_jesModified.csv
                        -   {sector_data_dir_path}/nlcd/NLCD_2012_forest_binary_gepa.parquet
                        -   {sector_data_dir_path}/forestlands_grasslands/mtbs_perims_DD.shp
                        -   {global_data_dir_path}/tl_2020_us_state/tl_2020_us_state.shp
                        -   {emi_data_dir_path}/forest_land_emi.csv

Output Files:           -   forest_land_proxy.parquet

Notes:                  -   This script assigns proxy GHGI emissions for forest land
                            remaining forest land using MTBS fire data.
                        -   The proxy geometries are brought in from the MTBS fire
                            permiter data. Some year-state combinations are missing
                            MTBS emissions data and are given a proportion of 1.0 for
                            the entire state. These state-years are given the
                            geometry of the forest land cover from the 2012 NLCD data.
"""

# %%
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
from pyarrow import parquet
import pytask
from pytask import Product, mark, task
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid
from ng_oil_production_utils import find_closest_year

from gch4i.config import (
    emi_data_dir_path,
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
from gch4i.utils import convert_FIPS_to_two_letter_code, normalize


# %%
def get_habitat_types(dataset, fuelbed_column, habitat_types):
    """
    Get the specific habitat types of interest from the MTBS data.

    Parameters:
    dataset (pd.DataFrame): The dataset to search for the habitat types of interest.
    fuelbed_column (str): The column name of the fuelbed to search for in the dataset.
    habitat_types (list): The list of habitat types to search for in the dataset.

    Returns:
    habitat_df (pd.Dataframe): A subset DataFrame of dataset that contains only the habitat types of interest.
    """

    habitat_df = dataset[
        dataset[fuelbed_column].str.lower().str.contains(habitat_types, na=False)
    ]

    return habitat_df


def calculate_state_emissions(emi_df, proxy_df, year_range):
    """
    Function to calculate emissions proxy for states based on the GHGI emissions
    data.

    Parameters:
    - emi_df: DataFrame containing GHGI emissions data.
    - proxy_df: DataFrame containing proxy data.
    - year_range: List or range of years to process.

    Returns:
    - final_proxy_df: DataFrame containing the processed emissions data for each
    state.
    """

    # Create the final dataframe that we will populate below
    final_proxy_df = pd.DataFrame()

    # Get the unique states in the emi data
    unique_states = emi_df["state_code"].unique()

    for year in year_range:

        # Filter proxy_df for the current year
        year_proxy = proxy_df[proxy_df["year"] == year].copy()

        # Process emissions for each state
        for state in unique_states:

            year_state_proxy = year_proxy[year_proxy["state_code"] == state].copy()

            # Filter out MTBS ch4 emissions that are 0 so they can be replaced with
            # a dummy row
            year_state_proxy = year_state_proxy[year_state_proxy["ch4_mg"] > 0]

            # if no data exist, skip it for now and we will fill it in later.
            if len(year_state_proxy) == 0:
                continue

            # Group by eventID and sum the ch4_mg and burnBndAc columns
            year_state_proxy = year_state_proxy.groupby(
                ["eventID"], as_index=False
            ).agg(
                {
                    "eventID": "first",
                    "ch4_mg": "sum",
                    "year": "first",
                    "state_code": "first",
                }
            )

            # Calculate the proportion of the total MTBS proxy emissions for the
            # year-state
            year_state_proxy.loc[:, "emissions"] = (
                year_state_proxy["ch4_mg"] / year_state_proxy["ch4_mg"].sum()
            )

            # Concatenate to the final dataframe
            final_proxy_df = pd.concat(
                [final_proxy_df, year_state_proxy], ignore_index=True
            )

    return final_proxy_df


def check_missing_data(emi_df, proxy_df):
    fires_by_state_year = (
        # forest_land_proxy.groupby(["state_code", "year"])
        # mtbs_burn_perimeters.groupby(["state_code", "year"])
        proxy_df.groupby(["state_code", "year"])
        .size()
        .rename("mtbs_fire_count")
    )
    check_state_year = emi_df.set_index(["state_code", "year"]).join(
        fires_by_state_year, how="left", rsuffix="_fires"
    )
    states_missing_fires = check_state_year[
        check_state_year["mtbs_fire_count"].isnull()
        & (check_state_year["ghgi_ch4_kt"] > 0)
    ].reset_index()
    return states_missing_fires


# %%
fccs_fuelbed_path = (
    sector_data_dir_path / "forestlands_grasslands/fccs_fuelbed_Aug2023_jesModified.csv"
)

nawfd_fuelbed_path = (
    sector_data_dir_path
    / "forestlands_grasslands/nawfd_fuelbed_Aug2023_jesModified.csv"
)
mtbs_burn_perimeter_path = (
    sector_data_dir_path / "forestlands_grasslands/mtbs_perims_DD.shp"
)

state_path = global_data_dir_path / "tl_2020_us_state.zip"


# %%

parameter_dict = dict(
    forest=dict(
        input_proxy_path=sector_data_dir_path
        / "forestlands_grasslands/MTBS_byEventFuelFuelbed_09Sep2024.csv",
        fccs_fuelbed_path=fccs_fuelbed_path,
        nawfd_fuelbed_path=nawfd_fuelbed_path,
        mtbs_burn_perimeter_path=mtbs_burn_perimeter_path,
        state_path=state_path,
        emi_path=emi_data_dir_path / "forest_land_emi.csv",
        nlcd_path=sector_data_dir_path / "nlcd/NLCD_2012_forest_binary_gepa.parquet",
        output_path=proxy_data_dir_path / "forest_land_proxy.parquet",
    ),
    grass=dict(
        input_proxy_path=sector_data_dir_path
        / "forestlands_grasslands/MTBS_byEventFuelFuelbed_09Sep2024.csv",
        fccs_fuelbed_path=fccs_fuelbed_path,
        nawfd_fuelbed_path=nawfd_fuelbed_path,
        mtbs_burn_perimeter_path=mtbs_burn_perimeter_path,
        state_path=state_path,
        emi_path=emi_data_dir_path / "grassland_emi.csv",
        nlcd_path=sector_data_dir_path / "nlcd/NLCD_2012_grass_binary_gepa.parquet",
        output_path=proxy_data_dir_path / "grassland_proxy.parquet",
    ),
)

(
    input_proxy_path,
    fccs_fuelbed_path,
    nawfd_fuelbed_path,
    mtbs_burn_perimeter_path,
    state_path,
    emi_path,
    nlcd_path,
    output_path,
) = parameter_dict["grass"].values()
# ) = parameter_dict["forest"].values()


# %% pytask function
def task_forest_grass_land_proxy_data(
    input_proxy_path: Path,
    fccs_fuelbed_path: Path,
    nawfd_fuelbed_path: Path,
    mtbs_burn_perimeter_path: Path,
    state_path: Path,
    emi_path: Path,
    nlcd_path: Path,
    output_path: Annotated[Path, Product],
) -> None:
    """
    This function processes the forest land proxy data and calculates the proxy emissions.

    Args:
    proxy_path: Path to the forest land proxy data.
    mtbs_lat_long_path: Path to the MTBS lat long data.
    fccs_fuelbed_path: Path to the FCCS fuelbed data.
    nawfd_fuelbed_path: Path to the NAWFD fuelbed data.
    state_path: Path to the state shapefile.
    output_path: Path to save the final proxy data.
    forest_land_emi: DataFrame containing the forest land emissions data.

    Returns:
    None. Proxy data is saved to a parquet file at the output_path.
    """
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
        .drop(columns=["state_name", "statefp"])
    )
    mtbs_burn_perimeters = (
        gpd.read_file(mtbs_burn_perimeter_path)
        .to_crs(4326)
        .assign(
            year_month=lambda df: df["Ig_Date"].dt.strftime("%Y-%m"),
            year=lambda df: df["Ig_Date"].dt.year,
            month=lambda df: df["Ig_Date"].dt.month,
        )
        .rename(columns={"Event_ID": "eventID", "Ig_Date": "event_date"})
        .loc[:, ["geometry", "year", "month", "year_month", "eventID"]]
        .assign(geometry=lambda df: df["geometry"].apply(make_valid))
        # .sjoin(state_gdf[["state_code", "geometry"]])
        .overlay(state_gdf, how="intersection")
        .query("year.isin(@years) & state_code.isin(@state_gdf.state_code)")
        .set_index("eventID")
    )
    mtbs_burn_perimeters

    if not mtbs_burn_perimeters["geometry"].is_valid.all():
        raise ValueError(f"Invalid geometries still present in the proxy data")
    # %%
    nlcd_gdf = (
        gpd.read_parquet(nlcd_path)
        .overlay(state_gdf, how="intersection")
        .rename(columns={"values": "ch4_mg"})
        # .dissolve(by="state_code", aggfunc="first")
        # .rename(columns={1: "rel_emi"})
    )
    nlcd_gdf
    # nlcd_gdf.plot(figsize=(10, 10))
    # %%
    # Read in the proxy data
    input_proxy_df = pd.read_csv(input_proxy_path)
    fccs_fuelbed = pd.read_csv(fccs_fuelbed_path)
    nawfd_fuelbed = pd.read_csv(nawfd_fuelbed_path)

    # Read in the emi data
    emi_df = pd.read_csv(emi_path, usecols=["state_code", "year", "ghgi_ch4_kt"]).query(
        "state_code.isin(@state_gdf.state_code)"
    )

    #  Step 1 - Data wrangling

    # Edit the fuelbed_aggregate column to remove the 'evg' string and convert to an
    # integer
    input_proxy_df["fuelbed_aggregate"] = (
        input_proxy_df["fuelbed_aggregate"]
        .str.replace("evg", "")
        .astype(float)
        .astype(int)
    )

    # Convert the FIPS state codes to two-letter state codes
    input_proxy_df = convert_FIPS_to_two_letter_code(input_proxy_df, "originstatecd")

    # %%

    # Get the forest and nonforest habitat types from the FCCS data
    fccs_forest = get_habitat_types(fccs_fuelbed, "FUELBED", "forest")

    nawfd_forest = get_habitat_types(nawfd_fuelbed, "name", "forest")
    # %%
    # Filter the MTBS data to only include forest habitat types in the FCCS and NAWFD data
    habitat_mask = input_proxy_df["fuelbed_aggregate"].isin(
        fccs_forest["FCCS"]
    ) | input_proxy_df["fuelbed_aggregate"].isin(nawfd_forest["nawfd_id"])
    filtered_proxy_df = input_proxy_df[habitat_mask]
    filtered_proxy_df

    other_fires_df = input_proxy_df[~habitat_mask]
    # %%
    #  Step 2 - Calculate proxy emissions for each state
    fire_emissions_sums_df = (
        filtered_proxy_df.groupby(["eventID"])
        .agg(
            {
                # "eventID": "first",
                "ch4_mg": "sum",
                # "year": "first",
                # "state_code": "first",
            }
        )
        .query("ch4_mg.gt(0)")
    )
    fire_emissions_sums_df
    # %%
    # Join the MTBS lat long data to the proxy data
    land_proxy_gdf = mtbs_burn_perimeters.join(fire_emissions_sums_df, how="inner")
    land_proxy_gdf

    # Check if there are any states missing fires
    states_missing_fires = check_missing_data(emi_df, land_proxy_gdf)
    states_missing_fires
    # %%

    # if we have states missing data, we try 3 things to fill them in:
    # 1. Fill in the missing data with the MTBS burn perimeters data that does not
    #    have a fuel / emissions data associated with it. The are ignored at first, but
    #    we can use them to fill in the missing data and assume equal allocation of
    #    their emissions across the year.
    # 2. If there are still states missing data, we will look for the closest year in
    #    the MTBS burn perimeters data and use that to fill in the missing data.
    # 3. If there are still states missing data, we will fill in the missing data with
    #    the NLCD data. We will create a new row for each month of the year
    if not states_missing_fires.empty:
        supp_data_df = pd.DataFrame()
        for row in states_missing_fires.itertuples():
            missing_state = row.state_code
            missing_year = row.year
            print(f"looking for data for state: {missing_state}, year: {missing_year}")
            mtbs_fill_in_gdf = mtbs_burn_perimeters.query(
                "(state_code == @missing_state) & (year == @missing_year)"
            ).assign(
                ch4_mg=1.0,
                # month=lambda df: [list(range(1, 13)) for _ in range(df.shape[0])],
            )
            if not mtbs_fill_in_gdf.empty:
                print(
                    f"\tFound data: {mtbs_fill_in_gdf.shape[0]} rows for state: {missing_state}, year: {missing_year}"
                )
                supp_data_df = pd.concat(
                    [supp_data_df, mtbs_fill_in_gdf], ignore_index=True
                )

        land_proxy_gdf = pd.concat([land_proxy_gdf, supp_data_df], ignore_index=True)
    states_missing_fires = check_missing_data(emi_df, land_proxy_gdf)
    states_missing_fires
    # %%
    if not states_missing_fires.empty:
        # If there are still states missing fires, we will look for the closest year
        # in the MTBS burn perimeters data and use that to fill in the missing data.
        supp_data_df = pd.DataFrame()
        for row in states_missing_fires.itertuples():
            missing_state = row.state_code
            missing_year = row.year
            print(f"looking for data for state: {missing_state}, year: {missing_year}")
            # first look if the state has any data in the MTBS burn perimeters regardless
            # of year.
            state_data = mtbs_burn_perimeters.query("state_code == @missing_state")
            if not state_data.empty:
                iyear_closest = find_closest_year(state_data.year, missing_year)
                print(
                    f"\tFound data for state: {missing_state} with year {iyear_closest}"
                )
                sup_data = state_data.query("year == @iyear_closest").assign(
                    year=missing_year,
                    ch4_mg=1.0,
                )
                supp_data_df = pd.concat([supp_data_df, sup_data], ignore_index=True)

        land_proxy_gdf = pd.concat([land_proxy_gdf, supp_data_df], ignore_index=True)
    states_missing_fires = check_missing_data(emi_df, land_proxy_gdf)
    states_missing_fires
    # %%
    # Merge the state data with the NLCD geometry data
    if not states_missing_fires.empty:
        # If there are still states missing fires, we will look for the closest year
        # in the MTBS burn perimeters data and use that to fill in the missing data.
        supp_data_df = pd.DataFrame()
        for row in states_missing_fires.itertuples():
            missing_state = row.state_code
            missing_year = row.year
            nlcd_fill_in_gdf = (
                nlcd_gdf.query("state_code == @missing_state")
                .assign(
                    year=missing_year,
                    month=lambda df: [list(range(1, 13)) for _ in range(df.shape[0])],
                )
                .explode("month")
                .assign(
                    year_month=lambda df: df["year"].astype(str)
                    + "-"
                    + df["month"].astype(str)
                )
            )
            if not nlcd_fill_in_gdf.empty:
                print(
                    f"\tFound data for state: {missing_state} with year {iyear_closest}"
                )
                supp_data_df = pd.concat(
                    [supp_data_df, nlcd_fill_in_gdf], ignore_index=True
                )

        # Concatenate the two dataframes
        land_proxy_gdf = pd.concat([land_proxy_gdf, supp_data_df], ignore_index=True)
    states_missing_fires = check_missing_data(emi_df, land_proxy_gdf)
    states_missing_fires

    if not states_missing_fires.empty:
        raise ValueError(
            f"Some states are still missing fires after filling in with NLCD data: {states_missing_fires}"
        )
    # %%
    # step 3 Create the final proxy dataframe

    land_proxy_gdf["annual_rel_emi"] = land_proxy_gdf.groupby(["state_code", "year"])[
        "ch4_mg"
    ].transform(normalize)
    # sum of the rel_emi = 1 for each state_code-year_month combination
    # used to allocate monthly emissions to monthly proxy
    land_proxy_gdf["rel_emi"] = land_proxy_gdf.groupby(["state_code", "year_month"])[
        "ch4_mg"
    ].transform(normalize)

    # we make sure the monthly scaling factors pass a quick check
    annual_all_eq_df = (
        land_proxy_gdf.groupby(["state_code", "year"])["annual_rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(
            is_close=lambda df: (
                np.isclose(df["sum_check"], 1, atol=0, rtol=0.00001)
                | np.isclose(df["sum_check"], 0, atol=0, rtol=0.00001)
            )
        )
    )
    annual_all_eq_df

    # we make sure the relative emissions for year_month pass
    if not annual_all_eq_df["is_close"].all():
        raise ValueError("not all annual values are normed correctly!")
    all_eq_df = (
        land_proxy_gdf.groupby(["state_code", "year_month"])["rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(
            is_close=lambda df: (
                np.isclose(df["sum_check"], 1, atol=0, rtol=0.00001)
                | np.isclose(df["sum_check"], 0, atol=0, rtol=0.00001)
            )
        )
    )
    all_eq_df

    if not all_eq_df["is_close"].all():
        raise ValueError("not all year_month values are normed correctly!")

    final_proxy_gdf = land_proxy_gdf[
        [
            "state_code",
            "year",
            "month",
            "year_month",
            "annual_rel_emi",
            "rel_emi",
            "geometry",
        ]
    ]
    ax = final_proxy_gdf.query("year == 2021").plot("state_code", figsize=(10, 10))
    ax.set(title=f"{output_path.name.split("_")[0]} proxy data for 2021")
    state_gdf.boundary.plot(ax=ax, color="black", linewidth=0.5)

    final_proxy_gdf.to_parquet(output_path, index=False)


# %%
sesh = pytask.Session(
    tasks=[
        task_forest_grass_land_proxy_data(**kwargs)
        for id, kwargs in parameter_dict.items()
    ]
)
# %%
