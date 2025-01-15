from pathlib import Path
from typing import Annotated
from zipfile import ZipFile
import calendar
import datetime

from pyarrow import parquet
import pandas as pd
import osgeo
import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark
import geopy
from geopy.geocoders import Nominatim
import duckdb

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
)

from gch4i.utils import name_formatter

t_to_kt = 0.001
year_range = [*range(min_year, max_year+1,1)] #List of emission years
year_range_str=[str(i) for i in year_range]
num_years = len(year_range)


@mark.persist
@task(id="oil_refineries_proxy")
def task_get_oil_refineries_proxy_data(
    subpart_y_path = "https://data.epa.gov/efservice/y_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_refineries_proxy.parquet",
):
    """
    Relative emissions and location information for reporting facilities are taken from 
    the GHGRP Subpart Y database.
    """

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

    # Reporting facilities from Subpart HH
    facility_df = (
    pd.read_csv(
        subpart_y_path,
        usecols=("facility_name",
                 "facility_id",
                 "reporting_year",
                 "ghg_quantity",
                 "latitude",
                 "longitude",
                 "state",
                 "zip"))
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"reporting_year": "year", "ghg_quantity": "ch4_t", "state": "state_code"})
    .drop_duplicates(subset=['facility_id', 'year'], keep='last')
    .astype({"year": int})
    .query("year.between(@min_year, @max_year)")
    .query("state_code.isin(@state_gdf['state_code'])")
    .reset_index(drop=True)
    )

    facility_df['rel_emi'] = facility_df.groupby(["state_code", "year"])['ch4_t'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    facility_df = facility_df.drop(columns='ch4_t')

    # Check that relative emissions sum to 1.0 each state/year combination
    sums = facility_df.groupby(["state_code", "year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"  # assert that the sums are close to 1

    facility_gdf = (
        gpd.GeoDataFrame(
            facility_df,
            geometry=gpd.points_from_xy(
                facility_df["longitude"],
                facility_df["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["latitude", "longitude", "zip"])
        .loc[:, ["facility_id", "facility_name", "state_code", "geometry", "year", "rel_emi"]]
    )

    facility_gdf.to_parquet(proxy_output_path)
    return None
