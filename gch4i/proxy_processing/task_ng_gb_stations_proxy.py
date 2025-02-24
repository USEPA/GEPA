"""
Name:                   task_ng_gb_stations_proxy.py
Date Last Modified:     2025-01-30
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Mapping of natural gas proxies.
Input Files:            State Geo: global_data_dir_path / "tl_2020_us_state.zip"
                        Enverus Midstream: sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb"
Output Files:           proxy_data_dir_path / "ng_gb_stations_proxy.parquet"
"""

# %%
from pathlib import Path
from typing import Annotated

import pandas as pd
import geopandas as gpd
import numpy as np
from pytask import Product, task, mark

from gch4i.config import (
    proxy_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    years,
)

from gch4i.utils import us_state_to_abbrev


# %% Pytask Function


@mark.persist
@task(id="ng_gb_stations_proxy")
def task_get_ng_gb_stations_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_midstream_ng_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb",
    gb_stations_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_gb_stations_proxy.parquet",
):
    """
    Creation of the following proxies using Enverus Midstream Rextag_Natural_Gas.gdb:
    - gb_stations_proxy - gathering compressor stations (NG Production)
    
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

    # Enverus Midstream Natural Gas Compressor Stations
    compressor_stations_gdf = (gpd.read_file(
        enverus_midstream_ng_path,
        layer="CompressorStations",
        columns=["NAME", "TYPE", "STATUS", "STATE_NAME", "CNTRY_NAME", "geometry"])

        .query("STATUS == 'Operational'")
        .query("CNTRY_NAME == 'United States'")
        .query("STATE_NAME.isin(@state_gdf['state_name'])")
        .drop(columns=["STATUS", "CNTRY_NAME"])
        .rename(columns={"NAME": "facility_name",
                         "TYPE": "type",
                         "STATE_NAME": "state_name",
                         })
        .assign(state_code='NaN')
        .assign(station_count=1.0)
        .to_crs(4326)
        .reset_index(drop=True)
        )

    for istation in np.arange(0, len(compressor_stations_gdf)):
        compressor_stations_gdf.loc[istation, "state_code"] = (
            us_state_to_abbrev(compressor_stations_gdf.loc[istation, "state_name"])
        )
    
    # gb_stations_proxy
    gb_stations_proxy_gdf = (
        compressor_stations_gdf
        .query("type == 'Gathering'")
        .drop(columns=["type", "state_name"])
        .loc[:, ["facility_name", "state_code", "geometry"]]
        .reset_index(drop=True)
        )

    # assume emissions are evenly distributed across the years and months
    monthly_proxy = pd.DataFrame()
    for iyear in years:
        temp_data_iyear = (compressor_stations_gdf.copy()
                           .assign(year=str(iyear))
                           )
        for imonth in range(1, 13):
            imonth_str = f"{imonth:02}"  # convert to 2-digit months
            year_month_str = str(iyear)+'-'+imonth_str
            temp_data_imonth = (temp_data_iyear
                                .assign(year_month=year_month_str)
                                .assign(month=imonth)
                                )
            monthly_proxy = pd.concat([monthly_proxy, temp_data_imonth]).reset_index(drop=True)

    # assign rel_emi and annual_rel_emi
    monthly_proxy['annual_rel_emi'] = monthly_proxy.groupby(['state_code', 'year'])['station_count'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    monthly_proxy['rel_emi'] = monthly_proxy.groupby(['state_code', 'year_month'])['station_count'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    monthly_proxy = monthly_proxy.drop(columns='station_count')

    # Check that annual relative emissions sum to 1.0 each state/year combination
    sums_annual = monthly_proxy.groupby(["state_code", "year"])["annual_rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums_annual, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums_annual}"  # assert that the sums are close to 1

    # Check that monthly relative emissions sum to 1.0 each state/year_month combination
    sums_monthly = monthly_proxy.groupby(["state_code", "year_month"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums_monthly, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year_month and state; {sums_monthly}"  # assert that the sums are close to 1

    monthly_proxy.to_parquet(gb_stations_output_path)

    return None
