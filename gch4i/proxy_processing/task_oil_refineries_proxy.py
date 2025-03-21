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
    proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_refineries_proxy.parquet",
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
                 .dropna(subset=["latitude", "longitude"])
                 .astype({'year': str})
                 .reset_index(drop=True)
                 )

    # Create relative emissions
    # sum of the rel_emi = 1 for each state_code-year_month combination
    # because the values in data_temp['rel_emi'] will be copied to each month, the 
    # the following normalization will lead to monthly totals = 1
    facility_df['rel_emi'] = (
        facility_df.groupby(["state_code", "year"])['ch4_t']
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
        )
    # sum of the annual_rel_emi = 1 for each state_code-year combination
    # because data_temp['rel_emi'] was copied to each month, rel_emi divided by 12 
    # will lead to annual totals = 1
    facility_df['annual_rel_emi'] = facility_df['rel_emi'] * 1/12
    # Drop ch4_t column
    facility_df = facility_df.drop(columns='ch4_t')

    # Create monthly proxy data
    monthly_proxy_data = pd.DataFrame()
    for imonth in range(1, 13):
        imonth_str = f"{imonth:02}"  # convert to 2-digit months
        data_temp_imonth = facility_df.copy()
        data_temp_imonth = data_temp_imonth.assign(year_month=lambda df: df['year']+'-'+imonth_str).assign(month=imonth)
        monthly_proxy_data = pd.concat([monthly_proxy_data, data_temp_imonth])

    # Check that relative emissions sum to 1.0 each state/year combination
    # get sums to check normalization
    sums = monthly_proxy_data.groupby(["state_code", "year"])["annual_rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Annual relative emissions do not sum to 1 for each year and state; {sums}"

    # Check that relative emissions sum to 1.0 each state/year combination
    # get sums to check normalization
    sums = monthly_proxy_data.groupby(["state_code", "year_month"])["rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Monthly relative emissions do not sum to 1 for each year and state; {sums}"

    # Create GeoDataFrame with geometries from lat & lon
    facility_gdf = (
        gpd.GeoDataFrame(
            monthly_proxy_data,
            geometry=gpd.points_from_xy(
                monthly_proxy_data["longitude"],
                monthly_proxy_data["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["facility_id", "facility_name", "latitude", "longitude", "zip"])
        .loc[:, ["year", "month", "year_month", "state_code", "annual_rel_emi", "rel_emi", "geometry"]]
        .astype({'year': int})
        .reset_index(drop=True)
    )

    # Save to parquet
    facility_gdf.to_parquet(proxy_output_path)
    return None

# %%
