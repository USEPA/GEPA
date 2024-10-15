# %%
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

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    max_year,
    min_year,
)

from gch4i.utils import us_state_to_abbrev

# %%
@mark.persist
@task(id="ng_compressor_stations_proxy")
def task_get_ng_compressor_stations_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_midstream_ng_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb",
    gb_stations_output_path: Annotated[Path, Product] = proxy_data_dir_path / "gb_stations_proxy.parquet",
    storage_comp_station_output_path: Annotated[Path, Product] = proxy_data_dir_path / "storage_comp_station_proxy.parquet",
    trans_comp_station_output_path: Annotated[Path, Product] = proxy_data_dir_path / "trans_comp_station_proxy.parquet",
):
    """
    Creation of the following proxies using Enverus Midstream Rextag_Natural_Gas.gdb:
    - gb_stations_proxy - gathering compressor stations (NG Production)
    - storage_comp_station_proxy - storage compressor stations (NG Storage)
    - trans_comp_station_proxy - transmission compressor stations (NG Transmission)
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
        .to_crs(4326)
        .reset_index(drop=True)
        )
    
    for istation in np.arange(0, len(compressor_stations_gdf)):
        compressor_stations_gdf.loc[istation, "state_code"] = us_state_to_abbrev(compressor_stations_gdf.loc[istation, "state_name"])
    
    # gb_stations_proxy
    gb_stations_proxy_gdf = (compressor_stations_gdf
                        .query("type == 'Gathering'")
                        .drop(columns=["type", "state_name"])
                        .loc[:, ["facility_name", "state_code", "geometry"]]
                        .reset_index(drop=True))
    gb_stations_proxy_gdf.to_parquet(gb_stations_output_path)

    # storage_comp_station_proxy
    storage_comp_station_proxy_gdf = (compressor_stations_gdf
                        .query("type == 'Storage'")
                        .drop(columns=["type", "state_name"])
                        .loc[:, ["facility_name", "state_code", "geometry"]]
                        .reset_index(drop=True))
    storage_comp_station_proxy_gdf.to_parquet(storage_comp_station_output_path)

    # trans_comp_station_proxy
    trans_comp_station_proxy_gdf = (compressor_stations_gdf
                        .query("type == 'Transmission'")
                        .drop(columns=["type", "state_name"])
                        .loc[:, ["facility_name", "state_code", "geometry"]]
                        .reset_index(drop=True))
    trans_comp_station_proxy_gdf.to_parquet(trans_comp_station_output_path)

    return None
