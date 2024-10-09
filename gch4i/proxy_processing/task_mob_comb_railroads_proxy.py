"""
Name:                   task_mob_comb_railroads_proxy.py
Date Last Modified:     2024-09-10
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of mobile combustion proxy emissions
Input Files:            -
Output Files:           -
Notes:                  - Script will be updated to remove comments once adjustments
                            are made for other mobile_combustion proxy emissions
                        -
"""

########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import numpy as np

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year
)

import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, GeometryCollection

########################################################################################
# %% STEP 0.2 Load Path Files
raw_path = Path(global_data_dir_path) / "raw"
Railroad_file_a = str(global_data_dir_path / "raw/tl_")
gdf_state_files = str(global_data_dir_path / "tl_2020_us_state/tl_2020_us_state.shp")

########################################################################################
# %% Define local variables
#start_year = 2012  # First year in emission timeseries
#end_year = 2022    # Last year in emission timeseries
year_range = [*range(min_year, max_year+1, 1)]  # List of emission years
year_range_str = [str(i) for i in year_range]
num_years = len(year_range)

########################################################################################
# %% Functions

def read_states():
    """
    Read in State spatial data
    """

    states_gdf = gpd.read_file(gdf_state_files)

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
        # Filter the collection to include only LineString or MultiLineString, Remove Points
        lines = [g for g in geom.geoms if isinstance(g, (LineString, MultiLineString))]
        if len(lines) == 1:
            return lines[0]
        elif len(lines) > 1:
            return MultiLineString(lines)
    else:
        return None  # Only returns LineStrings and MultiLineStrings (no polygons or points)

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
# %% Pytask


@mark.persist
@task(id="railroads_proxy")
def task_get_railroads_proxy(
    state_path: Path = gdf_state_files,
    reporting_proxy_output: Annotated[Path, Product] = proxy_data_dir_path
    / "mobile_combustion/railroads_proxy.parquet"
):
    """
    Relative location information for railroads in the US.
    """

    # Read in Railroad spatial data
    rail_list = []
    for year in year_range:
        rail_loc = (gpd.read_parquet(f"{Railroad_file_a}{year}_us_rails.parquet",
                                     columns=['MTFCC', 'geometry'])
                    .assign(year=year)
                    .drop(columns=['MTFCC']))
        rail_list.append(rail_loc)
    railroads = pd.concat(rail_list)
    railroads.to_crs("ESRI:102003", inplace=True)

    # Read in State spatial data
    states_gdf = read_states()

    # Join Railroad and State spatial data
    rail_list = []
    for year in year_range:
        rail_year = railroads[railroads['year'] == year]

        filtered_railroads = process_roads(rail_year, states_gdf)

        filtered_railroads = process_geometry_column(filtered_railroads)

        rail_list.append(filtered_railroads)

    railroads_filtered = pd.concat(rail_list)

    # Dissolve
    railroad_proxy = railroads_filtered.dissolve(by=['STUSPS', 'year']).reset_index()
    # Change column names
    railroad_proxy = railroad_proxy.rename(columns={'STUSPS': 'state_code',
                                                    'geometry': 'Geometry'})
    railroad_proxy = railroad_proxy.set_geometry('Geometry').to_crs("EPSG:4326")

    # Save output
    railroad_proxy.to_parquet(reporting_proxy_output)
    return None


########################################################################################
# Building function
########################################################################################

# def read_states():
#     """
#     Read in State spatial data
#     """

#     states_gdf = gpd.read_file(gdf_state_files)

#     states_gdf = states_gdf[~states_gdf['STUSPS'].isin(
#         ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
#         )]
#     states_gdf = states_gdf[['STUSPS', 'NAME', 'geometry']]
#     states_gdf = states_gdf.to_crs("ESRI:102003")

#     return states_gdf


# # Extract Line geometries from Geometry Collections
# def extract_lines(geom):
#     if geom is None:
#         return None
#     if isinstance(geom, (LineString, MultiLineString)):
#         # Return the geometry
#         return geom
#     elif isinstance(geom, GeometryCollection):
#         # Filter the collection to include only LineString or MultiLineString, Remove Points
#         lines = [g for g in geom.geoms if isinstance(g, (LineString, MultiLineString))]
#         if len(lines) == 1:
#             return lines[0]
#         elif len(lines) > 1:
#             return MultiLineString(lines)
#     else:
#         return None  # Only returns LineStrings and MultiLineStrings (no polygons or points)

# # Process geometry column
# def process_geometry_column(gdf):
#     # Apply the extract_lines function
#     gdf['geometry'] = gdf['geometry'].apply(extract_lines)

#     # Remove rows where geometry is None
#     gdf = gdf.dropna(subset=['geometry'])

#     # Ensure the GeoDataFrame only contains LineString and MultiLineString
#     gdf = gdf[gdf['geometry'].apply(lambda geom: isinstance(geom, (LineString, MultiLineString)))]

#     # Explode MultiLineStrings into LineStrings
#     gdf = gdf.explode(index_parts=True).reset_index(drop=True)

#     return gdf

# # %% Read in Railroad spatial data

# rail_parq_array = np.empty((num_years,), dtype=object)
# for iyear in np.arange(0, num_years):
#     rail_loc = gpd.read_parquet(Railroad_file_a +
#     year_range_str[iyear] + '_us_rails.parquet')

#     rail_loc['year'] = year_range_str[iyear]

#     rail_parq_array[iyear] = rail_loc

# rail_list = []
# for year in year_range:
#     rail_loc = (gpd.read_parquet(f"{Railroad_file_a}{year}_us_rails.parquet",
#                                     columns=['MTFCC', 'geometry'])
#                     .assign(year=year))
#     rail_list.append(rail_loc)
# railroads = pd.concat(rail_list)
# railroads.to_crs("ESRI:102003", inplace=True)

# railroads = railroads.dropna(subset=['geometry'])


# ######################################################################################
# # %% Read in State spatial data
# # gdf_states = gpd.read_file(gdf_state_files)

# # gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
# # ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
# # )]
# # gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]

# states_gdf = read_states()

# import time
# import geopandas as gpd
# import pandas as pd
# from tqdm import tqdm

# def process_roads(railroad_data, states_gdf):
#     start_time = time.time()
#     final_roads = []

#     # Process each state
#     for _, state in tqdm(states_gdf.iterrows(), desc="Processing states"):
#         # Create single state GeoDataFrame
#         state_gdf = gpd.GeoDataFrame(geometry=[state.geometry], crs=states_gdf.crs)

#         # Clip roads to state boundary
#         state_roads = gpd.clip(railroad_data, state_gdf)

#         if not state_roads.empty:
#             # Add state identifier
#             state_roads['STUSPS'] = state.STUSPS

#         final_roads.append(state_roads)
#         print(f"Processed {state.STUSPS}")

#     # Combine all results
#     final_result = pd.concat(final_roads)

#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"Total execution time: {execution_time:.2f} seconds")

#     return final_result

# rail_list = []
# for year in year_range:
#     rail_year = railroads[railroads['year'] == year]

#     filtered_railroads = process_roads(rail_year, gdf_states)

#     rail_list.append(filtered_railroads)

# railroads_filtered = pd.concat(rail_list)

# from shapely.geometry import MultiLineString, LineString, GeometryCollection

# railroads_filtered2 = process_geometry_column(railroads_filtered)

# dissolved_rail = railroads_filtered2.drop(columns=['MTFCC']).dissolve(by=['STUSPS', 'year']).reset_index()

# #
# import matplotlib.pyplot as plt


# rail_2012 = dissolved_rail[dissolved_rail['year'] == 2012]
# rail_2022 = dissolved_rail[dissolved_rail['year'] == 2022]
# rail_nc_2012 = rail_2012[rail_2012['STUSPS'] == 'NC']


# fig, ax = plt.subplots(figsize=(30, 45))

# #rail_2012.plot(ax=ax, color='green', linewidth=1)
# #rail_2022.plot(ax=ax, color='red', linewidth=1)
# rail_nc_2012.plot(ax=ax, color='blue', linewidth=1)
# states_gdf.boundary.plot(ax=ax, color='black', linewidth=1)

# plt.show()

# # # Alt method
# # rail_list = []
# # start_time = time.time()

# # buffer_distance = 10  # meters
# # states_gdf_buffer = states_gdf.copy()
# # states_gdf_buffer = states_gdf_buffer.buffer(buffer_distance)
# # states_gdf_buffer = gpd.GeoDataFrame(geometry=states_gdf_buffer, crs=states_gdf.crs)
# # states_gdf_buffer = states_gdf_buffer.join(states_gdf.drop(columns='geometry'))

# # for year in year_range:
# #     begin_time = time.time()

# #     rail_year = railroads[railroads['year'] == year]

# #     rail_split = gpd.overlay(rail_year, states_gdf, how='identity', keep_geom_type=False)

# #     # Step 3: Remove geometries that re not LineStrings or MultiLineStrings               # Check here if an issue
# #     #rail_split = process_geometry_column(rail_split)
# #     # Step 4: Use a state buffer to prevent accidental road reduction

# #     # Step 5 Spatial join to assign state attributes
# #     rails_with_states = gpd.sjoin(rail_split, states_gdf_buffer, how='left', predicate='within')

# #     rail_list.append(rails_with_states)

# #     end_time = time.time()
# #     execution_time = end_time - begin_time
# #     print(f"Total execution time: {execution_time:.2f} seconds")

# # railroads_filtered_alt = pd.concat(rail_list)
# # total_time = end_time - start_time
# # print(f"Total execution time: {total_time:.2f} seconds")
# # 3:53pm




# ######################################################################################
# # %% Join Railroad and State spatial data

# rail_state_array = np.empty((num_years,), dtype=object)
# for iyear in np.arange(0, num_years):
#     gdf_lines = rail_parq_array[iyear]

#     gdf_lines = gdf_lines.set_crs(gdf_states.crs)

#     rail_state_array[iyear] = gpd.sjoin(gdf_lines, gdf_states, how="left",
# predicate='within')
#     # Remove rows that are missing state designation
#     rail_state_array[iyear] = rail_state_array[iyear].dropna(subset=['STUSPS'])
#     # Keep only the columns of interest
#     rail_state_array[iyear] = rail_state_array[iyear][['year', 'STUSPS', 'geometry']]

# # %% Disolve and make one GeoDataFrame

# dissolved_list = []

# for iyear in range(len(rail_state_array)):
#     # Dissolve each GeoDataFrame by 'STUSPS' and 'year'
#     dissolved_gdf = rail_state_array[iyear].dissolve(by=['STUSPS', 'year'])

#     # Reset the index
#     dissolved_gdf = dissolved_gdf.reset_index()

#     # Append the dissolved GeoDataFrame to the list
#     dissolved_list.append(dissolved_gdf)

# # Concatenate all dissolved GeoDataFrames into one large GeoDataFrame
# railroad_proxy = gpd.GeoDataFrame(pd.concat(dissolved_list, ignore_index=True))

# # Change column names
# railroad_proxy = railroad_proxy.rename(columns={'STUSPS': 'state_code',
#                                                 'geometry': 'Geometry'})
# railroad_proxy = railroad_proxy.set_geometry('Geometry')


# # %% EXCESS TESTING

# import matplotlib.pyplot as plt

# # Plot the geometry
# railroad_proxy = railroad_proxy.set_geometry('Geometry')

# gdf_filtered = railroad_proxy[railroad_proxy['year'] == '2012']
# gdf_filtered = gdf_filtered[gdf_filtered['state_code'] == 'NY']

# fig, ax = plt.subplots(figsize=(10, 10))

# gdf_filtered.plot(ax=ax, color='blue', linewidth=1)

# plt.show()




# # %%
