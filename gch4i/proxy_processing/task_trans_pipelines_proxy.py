"""
Name:                   task_trans_pipelines_proxy.py
Date Last Modified:     2025-02-07
Authors Name:           John Bollenbacher (RTI International)
Purpose:                This file builds spatial proxies for natural gas pipelines on
                            farmlands.
                        Reads pipeline geometries, splits them by state, and saves out
                            a table of the resulting pipeline geometries for each year
                            and state.
Input Files:            - {sector_data_dir_path}/enverus/midstream/
                            Rextag_Natural_Gas.gdb
                        - {V3_DATA_PATH}/geospatial/cb_2018_us_state_500k/
                            cb_2018_us_state_500k.shp
Output Files:           - farm_pipelines_proxy.parquet
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import numpy as np
from pytask import Product, mark, task

from gch4i.config import (
    global_data_dir_path,
    max_year,
    min_year,
    proxy_data_dir_path,
    sector_data_dir_path
)


# %% Pytask Function
@mark.persist
@task(id="trans_pipeline_proxy")
def get_trans_pipeline_proxy_data(
    # Inputs
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    # Outputs
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "trans_pipelines_proxy.parquet"
    ),
):
    ###############################################################
    # Load data
    ###############################################################

    # Input path cannnot be in function because it is a directory, not a file
    enverus_midstream_pipelines_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb"

    # Load state shapes
    state_shapes = (
        gpd.read_file(state_path)
        .rename(columns={"STUSPS": "state_code"})
        .astype({"STATEFP": int})
        # get only lower 48 + DC
        .query("(STATEFP < 60) & (STATEFP != 2) & (STATEFP != 15)")
        .loc[:, ["state_code", "geometry"]]
        .to_crs(epsg=4326)
    )

    # load enverus pipelines geometries data
    enverus_pipelines_gdf = gpd.read_file(
        enverus_midstream_pipelines_path, layer='NaturalGasPipelines')
    # Ensure enverus_pipelines_gdf is in EPSG:4326
    enverus_pipelines_gdf = enverus_pipelines_gdf.to_crs(epsg=4326)

    # drop unneeded pipelines and columns
    # enverus_pipelines_gdf = enverus_pipelines_gdf[enverus_pipelines_gdf['CNTRY_NAME']=='United States'] #dropped, because not used in v2 and state intersection will filter this.
    enverus_pipelines_gdf = enverus_pipelines_gdf[enverus_pipelines_gdf['STATUS'] == 'Operational']
    enverus_pipelines_gdf = enverus_pipelines_gdf[enverus_pipelines_gdf['TYPE'] == 'Transmission']
    enverus_pipelines_gdf = enverus_pipelines_gdf[['LOC_ID',
                                                   'DIAMETER',
                                                   'geometry'  # ,'INSTALL_YR',
                                                   ]].reset_index(drop=True)

    # Split pipelines at state boundaries and keep only the segments that fall within states
    enverus_pipelines_gdf_split_by_state = gpd.overlay(
        enverus_pipelines_gdf, state_shapes, how='intersection', keep_geom_type=True)

    # create each year's proxy, and combine to make the full proxy table
    # NOTE: this assumes the geometries do not change year to year, since we dont have
    # good data for when each pipeline began operation.
    year_gdfs = []
    for year in range(min_year, max_year+1):
        year_gdf = enverus_pipelines_gdf_split_by_state.copy()
        # year_gdf = year_gdf[year_gdf['INSTALL_YR'] > year] # not applicable since most INSTALL_YR are NaN
        year_gdf['year'] = year
        year_gdfs.append(year_gdf)
    proxy_gdf = pd.concat(year_gdfs, ignore_index=True)

    # compute the length of each pipeline within each state
    proxy_gdf = proxy_gdf.to_crs("ESRI:102003")  # Convert to ESRI:102003
    proxy_gdf['pipeline_length_within_state'] = proxy_gdf.geometry.length
    proxy_gdf = proxy_gdf.to_crs(epsg=4326)  # Convert back to EPSG:4326

    # get rel_emi, simply assume rel_emi is proportional to length.
    proxy_gdf = proxy_gdf.rename(columns={'pipeline_length_within_state': 'rel_emi'})

    ###############################################################
    # Normalize and save
    ###############################################################

    # Normalize relative emissions to sum to 1 for each year and state
    # drop state-years with 0 total volume
    proxy_gdf = proxy_gdf.groupby(['year']).filter(lambda x: x['rel_emi'].sum() > 0)
    # normalize to sum to 1
    proxy_gdf['rel_emi'] = proxy_gdf.groupby(
        ['year'])['rel_emi'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    # get sums to check normalization
    sums = proxy_gdf.groupby(["year"])["rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Save to parquet
    proxy_gdf.to_parquet(output_path)

    return proxy_gdf

# proxy_gdf = get_trans_pipeline_proxy_data()

# #%%
# proxy_gdf.info()
# proxy_gdf
