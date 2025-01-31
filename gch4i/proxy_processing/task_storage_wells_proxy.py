"""
Name:                  task_storage_wells_proxy.py
Date Last Modified:    2025-01-23
Authors Name:          John Bollenbacher (RTI International)
Purpose:               Creates proxy data for storage wells using EIA field data
Input Files:           - EIA StoreFields Input: {sector_data_dir_path}/lng/
                        191 Field Level Storage Data (Annual).csv
                       - EIA StorFields Locs: {sector_data_dir_path}/lng/
                        EIA_Natural_Gas_Underground_Storage.csv
Output Files:          - {proxy_data_dir_path}/storage_wells_proxy.parquet
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import numpy as np
from pytask import Product, mark, task

from gch4i.config import (
    proxy_data_dir_path,
    sector_data_dir_path,
    V3_DATA_PATH,
)


# %% Pytask Function
@mark.persist
@task(id="storage_wells_proxy")
def get_storage_wells_proxy_data(
    # Inputs
    EIA_StorFields_inputfile: Path = sector_data_dir_path / 'lng/191 Field Level Storage Data (Annual).csv',
    EIA_StorFields_locs_inputfile: Path = sector_data_dir_path / 'lng/EIA_Natural_Gas_Underground_Storage.csv',

    # Outputs
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "storage_wells_proxy.parquet"
    ),
):
    """
    Create proxy data for storages wells using EIA field data.

    Args:
        EIA_StorFields_inputfile (Path): Path to EIA storage fields data.
        EIA_StorFields_locs_inputfile (Path): Path to EIA storage fields locations data.
        output_path (Path): Path to save the output proxy data.

    Returns:
        None: Writes the output proxy data to the specified path.
    """
    ################################################
    # Load and merge facilities and location data
    ################################################

    # load EIA Storage Field Capacities
    EIA_StorFields = pd.read_csv(EIA_StorFields_inputfile, skiprows=0)
    EIA_StorFields.columns = [c.replace('<BR>', ' ').strip() for c in EIA_StorFields.columns]
    EIA_StorFields = EIA_StorFields[['Year', 'Gas Field Code', 'Report State', 'Status',
                                     'Reservoir Code', 'Total Field Capacity(Mcf)',
                                     'County Name']]
    # Clean up county names
    EIA_StorFields['County Name'] = (EIA_StorFields['County Name']
                                     .str.replace(' County', '')
                                     .str.replace(' Parish', '')
                                     .str.replace(' Municipality', ''))
    # filter for active storage fields only
    EIA_StorFields = EIA_StorFields[EIA_StorFields['Status'] == 'Active']
    # rename columns
    EIA_StorFields = (EIA_StorFields.rename(columns={'Gas Field Code': 'Field Code'})
                                    .reset_index(drop=True))

    # load field locations
    EIA_StorFields_locs = pd.read_csv(EIA_StorFields_locs_inputfile, skiprows=0)
    EIA_StorFields_locs = EIA_StorFields_locs[['Field Code',
                                               'Reservoir Code',
                                               'Longitude',
                                               'Latitude']]
    EIA_StorFields_locs = EIA_StorFields_locs.reset_index(drop=True)

    # merge EIA storage fields and locations data
    merged_fields = EIA_StorFields.merge(EIA_StorFields_locs,
                                         on=['Field Code', 'Reservoir Code'],
                                         how='left')

    # Create a GeoDataFrame from the export terminals data
    proxy_gdf = gpd.GeoDataFrame(
        merged_fields,
        geometry=gpd.points_from_xy(merged_fields['Longitude'],
                                    merged_fields['Latitude']),
        crs="EPSG:4326"
    ).rename(columns={
        'Report State': 'state_code',
        'Year': 'year',
        'Total Field Capacity(Mcf)': 'rel_emi'
    })[['year',
        'state_code',
        'rel_emi',
        'geometry',
        'Field Code',
        'Reservoir Code',
        'County Name']].dropna(subset=['year', 'state_code', 'rel_emi'])

    ###############################################################
    # Use county geometry for facilities with no lat-lon
    ###############################################################

    # load county geometries as fallback locations for facilities without known lat-lon
    county_fips = pd.read_csv(V3_DATA_PATH / 'geospatial/county_fips.csv')[['STATEFP', 'COUNTYFP', 'STATE', 'COUNTYNAME']]
    county_shapes = gpd.read_file(V3_DATA_PATH / 'geospatial/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
    county_shapes['STATEFP'] = county_shapes['STATEFP'].astype('int64')
    county_shapes['COUNTYFP'] = county_shapes['COUNTYFP'].astype('int64')
    # merge county geometries with fips codes
    county_shapes = (county_shapes
                     .merge(county_fips,
                            how='left',
                            on=['STATEFP', 'COUNTYFP'])[['COUNTYNAME',
                                                         'STATE',
                                                         'geometry']]
                     .dropna())
    # clean up county names
    county_shapes['COUNTYNAME'] = (county_shapes['COUNTYNAME']
                                   .str.replace(' County', '')
                                   .str.replace(' Parish', '')
                                   .str.replace(' Municipality', ''))
    # rename columns
    county_shapes = county_shapes.rename(columns={'COUNTYNAME': 'County Name',
                                                  'STATE': 'state_code'})

    # manually rename counties which are improperly named in the facilities data
    proxy_gdf.loc[proxy_gdf['County Name'] == 'Bristo', 'County Name'] = 'Bristol'

    # Handle missing geometries
    missing_mask = proxy_gdf.geometry.apply(lambda x: x.is_empty)
    if missing_mask.any():
        missing_geom = proxy_gdf[missing_mask]
        merged_missing_geom = missing_geom.merge(
            county_shapes,
            on=['County Name', 'state_code'],
            how='left'
        )
        proxy_gdf.loc[missing_mask, 'geometry'] = merged_missing_geom['geometry_y'].values

    # drop the one remaining facility with no location or county. its capacity is very small, so negligible
    proxy_gdf = (proxy_gdf[~((proxy_gdf['year'] == 2018) &
                 (proxy_gdf['state_code'] == 'MS') &
                 (proxy_gdf['Field Code'] == 4084) &
                 (proxy_gdf['Reservoir Code'] == 1))])

    # drop now-unnecessary columns
    proxy_gdf = proxy_gdf[['year', 'state_code', 'rel_emi',
                           'geometry', 'Field Code', 'Reservoir Code']]

    ###############################################################
    # Normalize and save
    ###############################################################

    # Normalize relative emissions to sum to 1 for each year and state
    # drop state-years with 0 total volume
    proxy_gdf = (proxy_gdf.groupby(['state_code', 'year'])
                 .filter(lambda x: x['rel_emi'].sum() > 0))
    # normalize to sum to 1
    proxy_gdf['rel_emi'] = (proxy_gdf.groupby(['year', 'state_code'])['rel_emi']
                            .transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
    # get sums to check normalization
    sums = proxy_gdf.groupby(["state_code", "year"])["rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Save to parquet
    proxy_gdf.to_parquet(output_path)

    return None
