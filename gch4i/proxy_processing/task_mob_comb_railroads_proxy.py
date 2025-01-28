"""
Name:                   task_mob_comb_railroads_proxy.py
Date Last Modified:     2025-01-24
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of mobile combustion railroad proxy emissions
Input Files:            - Raw Input: {global_data_dir_path}/raw/
                        - Railroad Files: {global_data_dir_path}/raw/
                            tl_{year}_us_rails.parquet
                        - State Geo: {global_data_dir_path}/tl_2020_us_state.zip
Output Files:           - {proxy_data_dir_path}/railroads_proxy.parquet
"""

########################################################################################
# %% Load Packages

from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd

from gch4i.config import (
    proxy_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year
)

import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, GeometryCollection


########################################################################################
# %% Define variable constants
year_range = [*range(min_year, max_year+1, 1)]


########################################################################################
# %% Functions


def read_states(state_path):
    """
    Read in State spatial data
    """

    states_gdf = gpd.read_file(state_path)

    states_gdf = states_gdf[~states_gdf['STUSPS'].isin(
        ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
        )]
    states_gdf = states_gdf[['STUSPS', 'NAME', 'geometry']]
    states_gdf = states_gdf.to_crs("ESRI:102003")

    return states_gdf


# Extract Line geometries from Geometry Collections
def extract_lines(geom):
    if geom is None:
        return None
    if isinstance(geom, (LineString, MultiLineString)):
        # Return the geometry
        return geom
    elif isinstance(geom, GeometryCollection):
        # Filter the collection to include only LineString or MultiLineString
        lines = [g for g in geom.geoms if isinstance(g, (LineString, MultiLineString))]
        if len(lines) == 1:
            return lines[0]
        elif len(lines) > 1:
            return MultiLineString(lines)
    else:
        return None


# Process geometry column
def process_geometry_column(gdf):
    # Apply the extract_lines function
    gdf['geometry'] = gdf['geometry'].apply(extract_lines)

    # Remove rows where geometry is None
    gdf = gdf.dropna(subset=['geometry'])

    # Ensure the GeoDataFrame only contains LineString and MultiLineString
    gdf = gdf[gdf['geometry'].apply(lambda geom: isinstance(geom, (LineString, MultiLineString)))]

    # Explode MultiLineStrings into LineStrings
    gdf = gdf.explode(index_parts=True).reset_index(drop=True)

    return gdf


def process_roads(railroad_data, states_gdf):
    final_roads = []

    # Process each state
    for _, state in states_gdf.iterrows():
        # Create single state GeoDataFrame
        state_gdf = gpd.GeoDataFrame(geometry=[state.geometry], crs=states_gdf.crs)

        # Clip roads to state boundary
        state_roads = gpd.clip(railroad_data, state_gdf)

        if not state_roads.empty:
            # Add state identifier
            state_roads['STUSPS'] = state.STUSPS

        final_roads.append(state_roads)

    # Combine all results & Deduplicate
    final_result = pd.concat(final_roads)
    final_result = process_geometry_column(final_result)
    final_result = final_result.drop_duplicates(subset=['geometry'])

    return final_result

########################################################################################
# %% Pytask Function


@mark.persist
@task(id="railroads_proxy")
def task_get_railroads_proxy(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    reporting_proxy_output: Annotated[Path, Product] = proxy_data_dir_path
    / "railroads_proxy.parquet"
):
    """
    Process railroad geometries by state. Multilinestring geometries are exploded into
    individual linestrings and clipped to state boundaries.

    `raw_path` is the path to raw railroad data; however, it is not included in the
    pytask function arguments because it is not a file, but a directory of files

    Args:
        state_path: GeoDattaFrame containing state geometries.

    Returns:
        reporting_proxy_output: GeoDataFrame containing processed railroad geometries.
    """

    # Raw Railroad Data Path
    raw_path = global_data_dir_path / "raw"

    # Read in Railroad spatial data
    # Initialize railroad list
    rail_list = []
    # Read in year by year
    for year in year_range:
        rail_loc = (gpd.read_parquet(f"{raw_path}/tl_{year}_us_rails.parquet",
                                     columns=['MTFCC', 'geometry'])
                    .assign(year=year)
                    .drop(columns=['MTFCC']))
        # Append data to railroad list
        rail_list.append(rail_loc)
    # Concatenate all years
    railroads = pd.concat(rail_list)
    # Convert to CRS 102003 for processing
    railroads.to_crs("ESRI:102003", inplace=True)

    # Read in State spatial data
    states_gdf = read_states(state_path)

    # Join Railroad and State spatial data
    # Initialize railroad list
    rail_list = []
    # Read in year by year
    for year in year_range:
        # Filter by year
        rail_year = railroads[railroads['year'] == year]
        # Process roads (Join with state data, explode out multi-geometries, etc.)
        filtered_railroads = process_roads(rail_year, states_gdf)

        # Append data to railroad list
        rail_list.append(filtered_railroads)
    # Concatenate all years
    railroads_filtered = pd.concat(rail_list)

    # Dissolve
    railroad_proxy = railroads_filtered.dissolve(by=['STUSPS', 'year']).reset_index()
    # Change column names
    railroad_proxy = railroad_proxy.rename(columns={'STUSPS': 'state_code'})
    # Set geometry: 4326
    railroad_proxy = railroad_proxy.set_geometry('geometry').to_crs("EPSG:4326")

    # Save output
    railroad_proxy.to_parquet(reporting_proxy_output)
    return None
