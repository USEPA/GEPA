"""
Name:                   task_ng_well_blowout_proxy.py
Date Last Modified:     2025-07-01
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Mapping of natural gas storage well blowout proxy emissions in 
                        the Natural Gas Transmission and Storage segment.
Input Files:            None
Output Files:           proxy_data_dir_path / "ng_storage_well_blowout_proxy.parquet"
"""

# %%
import calendar
import datetime
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import osgeo
import pandas as pd
import seaborn as sns
from pyarrow import parquet
from pytask import Product, mark, task

from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    proxy_data_dir_path,
)
from gch4i.utils import name_formatter

# %%


@mark.persist
@task(id="ng_storage_well_blowout_proxy")
def task_get_ng_storage_well_blowout_proxy_data(
    output_path: Annotated[Path, Product] = (proxy_data_dir_path / "ng_storage_well_blowout_proxy.parquet"),
):
    """
    Storage well blowout events occur two times over 2012-2022. These locations and emissions are
    provided directly by the GHGI sector leads and manually coded into the proxy.

        1. CA in 2015
           state_code: CA; year: 2015; emi: 78.350 kt; lat: 34.31307; lon: -118.56462
        2. CA in 2016
           state_code: CA; year: 2015; emi: 78.350 kt; lat: 34.31307; lon: -118.56462
    """

    storage_well_blowout_df = pd.DataFrame(
        {'state_code': ['CA', 'CA'],
         'year': [2015, 2016],
         'rel_emi': [1.0, 1.0],
         'lat': [34.31307, 34.31307],
         'lon': [-118.56462, -118.56462],
    })

    storage_well_blowout_gdf = (gpd.GeoDataFrame(
        storage_well_blowout_df,
        geometry=gpd.points_from_xy(
            storage_well_blowout_df["lon"],
            storage_well_blowout_df["lat"],
            crs=4326
            )
        )
        .drop(columns=["lat", "lon"])
        .loc[:, ["year", "state_code", "geometry", "rel_emi"]]
    )

    # Check that relative emissions sum to 1.0 each state/year combination
    annual_sums = storage_well_blowout_gdf.groupby(["state_code", "year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(annual_sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {annual_sums}"  # assert that the sums are close to 1

    # Output Proxy Parquet Files
    storage_well_blowout_gdf = storage_well_blowout_gdf.astype({'year': int})
    storage_well_blowout_gdf.to_parquet(output_path)

    return None

# %%
