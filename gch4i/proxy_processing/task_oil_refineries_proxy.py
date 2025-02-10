"""
Name:                   task_oil_refineries_proxy.py
Date Last Modified:     2025-01-24
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Process oil refineries proxy data for methane emissions
Input Files:            - https://data.epa.gov/efservice/y_subpart_level_information/
                            pub_dim_facility/ghg_name/=/Methane/CSV"
                        - State Geodata: {global_data_dir_path}/tl_2020_us_state.zip
Output Files:           - {proxy_data_dir_path}/oil_refineries_proxy.parquet
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated
import pandas as pd
import geopandas as gpd
import numpy as np
from pytask import Product, task, mark

from gch4i.config import (
    proxy_data_dir_path,
    global_data_dir_path,
    min_year,
    max_year,
)


# %% Pytask Function
@mark.persist
@task(id="oil_refineries_proxy")
def task_get_oil_refineries_proxy_data(
    subpart_y_path="https://data.epa.gov/efservice/y_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    proxy_output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "oil_refineries_proxy.parquet"
    ),
):
    """
    Process oil refineries proxy data for methane emissions

    Args:
        subpart_y_path (str): URL to Subpart Y data
        state_path (Path): Path to state geodata

    Returns:
        None. Saves processed data to proxy_output_path
    """

    # Load in State ANSI data
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

    # Reporting facilities from Subpart Y
    facility_df = (pd.read_csv(
        subpart_y_path,
        usecols=("facility_name",
                 "facility_id",
                 "reporting_year",
                 "ghg_quantity",
                 "latitude",
                 "longitude",
                 "state",
                 "zip"))
                 # make columns lowercase
                 .rename(columns=lambda x: str(x).lower())
                 # rename columns
                 .rename(columns={"reporting_year": "year",
                                  "ghg_quantity": "ch4_t",
                                  "state": "state_code"})
                 # drop duplictes of facility_id and year, keeping the last
                 .drop_duplicates(subset=['facility_id', 'year'], keep='last')
                 # Select only years of interest
                 .query("year.between(@min_year, @max_year)")
                 # Select only facilities in the lower 48 + DC
                 .query("state_code.isin(@state_gdf['state_code'])")
                 .reset_index(drop=True)
                 )

    # Create Relative Emissions
    facility_df['rel_emi'] = (
        facility_df.groupby(["state_code", "year"])['ch4_t']
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
        )
    # Drop ch4_t column
    facility_df = facility_df.drop(columns='ch4_t')

    # Check that relative emissions sum to 1.0 each state/year combination
    # get sums to check normalization
    sums = facility_df.groupby(["state_code", "year"])["rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Create GeoDataFrame with geometries from lat & lon
    facility_gdf = (
        gpd.GeoDataFrame(
            facility_df,
            geometry=gpd.points_from_xy(
                facility_df["longitude"],
                facility_df["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["facility_id", "facility_name", "latitude", "longitude", "zip"])
        .loc[:, ["year", "state_code", "rel_emi", "geometry"]]
    )

    # Save to parquet
    facility_gdf.to_parquet(proxy_output_path)
    return None

# %%
