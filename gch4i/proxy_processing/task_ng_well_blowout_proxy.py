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
@task(id="ng_well_blowout_proxy")
def task_get_ng_well_blowout_proxy_data(
    output_path: Annotated[Path, Product] = (proxy_data_dir_path / "ng_well_blowout_proxy.parquet"),
):
    """
    Well blowouts occur three times over 2012-2022. These locations and emissions are
    provided directly by the GHGI sector leads and manually coded into the proxy.

        1. LA in 2019
           state_code: LA; year: 2019; emi: 49 kt; lat: 32.1; lon: -93.4
        2. OH in 2018
           state_code: OH; year: 2018; emi: 60 kt; lat: 39.864; lon: -80.861
        3. TX 2019
           state_code: TX; year: 2019; emi: 4.8 kt; lat: 28.9; lon: -97.6
    
    """

    well_blowout_df = pd.DataFrame(
        {'state_code': ['LA', 'OH', 'TX'],
         'year': [2019, 2018, 2019],
         'rel_emi': [1.0, 1.0, 1.0],  # assign 100% of the emissions to each state/year combination
         'lat': [32.1, 39.864, 28.9],
         'lon': [-93.4, -80.861, -97.6],
    })

    well_blowout_gdf = (gpd.GeoDataFrame(
        well_blowout_df,
        geometry=gpd.points_from_xy(
            well_blowout_df["lon"],
            well_blowout_df["lat"],
            crs=4326
            )
        )
        .drop(columns=["lat", "lon"])
        .loc[:, ["year", "state_code", "rel_emi", "geometry"]]
    )

    # Check that relative emissions sum to 1.0 each state/year combination
    sums = well_blowout_gdf.groupby(["state_code", "year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"  # assert that the sums are close to 1

    # Output Proxy Parquet Files
    well_blowout_gdf = well_blowout_gdf.astype({'year':str})
    well_blowout_gdf.to_parquet(output_path)

    return None
