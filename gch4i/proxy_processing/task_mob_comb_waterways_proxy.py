"""
Name:                   task_mob_comb_waterways_proxy.py
Date Last Modified:     2025-01-24
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of mobile combustion waterways proxy emissions
Input Files:            - Waterways: {global_data_dir_path}/raw/bts_2023_waterways.gpkg
                        - State Geo: {global_data_dir_path}/tl_2020_us_state.zip
Output Files:           - {proxy_data_dir_path}/waterways_proxy.parquet
"""

########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

from gch4i.config import (
    # V3_DATA_PATH,
    emi_data_dir_path,
    proxy_data_dir_path,
    global_data_dir_path,
    tmp_data_dir_path,
    max_year,
    min_year
)

########################################################################################
# %% Define local variables
year_range = [*range(min_year, max_year+1, 1)]  # List of emission years

########################################################################################
# %% Functions


def remove_z(geom):
    if geom is None:
        return None  # Handle None geometries
    if geom.has_z:
        # Handle LineString and MultiLineString separately
        if isinstance(geom, LineString):
            # Extract only X and Y from each point in the LineString
            return LineString([(x, y) for x, y, z in geom.coords])
        elif isinstance(geom, MultiLineString):
            # Apply the same logic for each LineString in MultiLineString
            return MultiLineString([LineString([(x, y) for x, y, z in line.coords]) for line in geom.geoms])
    return geom


# Read in State Spatial Data
def read_states(state_path):
    """
    Read in State spatial data
    """

    gdf_states = gpd.read_file(state_path)

    gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
        ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
        )]
    gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]
    gdf_states = gdf_states.to_crs("ESRI:102003")

    return gdf_states


def process_waterways(waterways_data, state_path):
    """
    Filter and clip waterways data to state boundaries
    """
    states_gdf = read_states(state_path)
    final_roads = []

    # Process each state
    for _, state in states_gdf.iterrows():
        # Create single state GeoDataFrame
        state_gdf = gpd.GeoDataFrame(geometry=[state.geometry], crs=states_gdf.crs)

        # Clip roads to state boundary
        state_roads = gpd.clip(waterways_data, state_gdf)

        if not state_roads.empty:
            # Add state identifier
            state_roads['STUSPS'] = state.STUSPS

        final_roads.append(state_roads)

    # Combine all results
    final_result = pd.concat(final_roads)

    return final_result


########################################################################################
# %% Pytask

@mark.persist
@task(id="waterways_proxy")
def task_get_waterways_proxy(
    waterways: Path = global_data_dir_path / "raw/bts_2023_waterways.gpkg",
    emi_data: Path = emi_data_dir_path / "emi_waterways.csv",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    reporting_proxy_output: Annotated[Path, Product] = tmp_data_dir_path
    / "waterways_proxy.parquet"
):
    """
    Relative location information for waterways in the US.

    Args:
        waterways: Path to waterways proxy data.
        emi_data: Path to waterways emissions data.
        state_path: Path to state geometries.
        reporting_proxy_output: Path to save output data.

    Returns:
        None. Saves output data to reporting_proxy_output.
    """
    # Read in waterways data
    waterways_loc = gpd.read_file(waterways)

    # Clean Proxy Data
    waterways_loc = (
        waterways_loc.filter(items=['geometry'])
        # Remove Z coordinates
        .assign(geometry=lambda x: x['geometry'].apply(remove_z))
        # Convert to EPSG:102003 for clipping
        .to_crs("ESRI:102003")
        # Explode MultiLineString geometries
        .explode()
        # Drop duplicate geometries
        .drop_duplicates(subset=['geometry'])
        )

    # Filter and Clip Waterways Data at State boundaries
    proxy_gdf = process_waterways(waterways_loc, state_path)

    # Clean proxy data
    proxy_gdf = (
        # Explode again to ensure no MultiLineString geometries
        proxy_gdf.explode()
        # Drop Duplicate geometries
        .drop_duplicates(subset=['geometry'])
        # Convert to EPSG:4326
        .to_crs("EPSG:4326")
        # Dissolve by state
        .dissolve(by=['STUSPS'])
        .reset_index()
        # Rename state_code column
        .rename(columns={'STUSPS': 'state_code'})
        # Set geometry column
        .set_geometry('geometry')
    )

    """
    The proxy data source is missing proxy data for certain states with emissions data:
        AZ, CO, ND, NM, NV, UT, WY

    To address this, we will add alternative data for these states by generalizing
    the emissions data to the state level. Future versions of this proxy should explore
    alternative proxy data sources. Consider TigerLines data source used for roads and
    railroads proxies.
    """
    # Read in emi data
    emi_df = pd.read_csv(emi_data)
    # Filter to CONUS states
    emi_df = emi_df.query("state_code not in ['AK', 'HI']")
    # Get unique state codes
    emi_states = set(emi_df['state_code'])
    proxy_states = set(proxy_gdf['state_code'])
    # Identify missing states
    missing_states = emi_states.difference(proxy_states)

    # Add missing states alternative data to proxy data
    if missing_states:
        # Create State GeoDataFrame
        alt_proxy = read_states(state_path)
        # Filter for state_codes in missing states
        alt_proxy = alt_proxy[alt_proxy['STUSPS'].isin(missing_states)]
        # Rename columns
        alt_proxy = alt_proxy.rename(columns={'STUSPS': 'state_code'})
        # Drop NAME column
        alt_proxy = alt_proxy.drop(columns=['NAME'])
        # Convert to EPSG:4326
        alt_proxy = alt_proxy.to_crs("EPSG:4326")
        # Append to proxy data
        proxy_gdf = pd.concat([proxy_gdf, alt_proxy], ignore_index=True)

    # Save output
    proxy_gdf.to_parquet(reporting_proxy_output)
    return None
