"""
Name:                   task_mob_comb_waterways_proxy.py
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

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    tmp_data_dir_path,
    max_year,
    min_year
)

########################################################################################
# %% STEP 0.2 Load Path Files

gdf_state_files = str(global_data_dir_path / "tl_2020_us_state/tl_2020_us_state.shp")
Waterways_file = str(global_data_dir_path / "raw" / "bts_2023_waterways.gpkg")


########################################################################################
# %% Define local variables
# start_year = 2012  # First year in emission timeseries
# end_year = 2022    # Last year in emission timeseries
year_range = [*range(min_year, max_year+1, 1)]  # List of emission years
year_range_str = [str(i) for i in year_range]
num_years = len(year_range)

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
def read_states():
    """
    Read in State spatial data
    """

    gdf_states = gpd.read_file(gdf_state_files)

    gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
        ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
        )]
    gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]
    gdf_states = gdf_states.to_crs("ESRI:102003")

    return gdf_states


def process_roads(waterways_data):
    """
    Filter and clip waterways data to state boundaries
    """
    states_gdf = read_states()
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
    state_path: Path = gdf_state_files,
    reporting_proxy_output: Annotated[Path, Product] = proxy_data_dir_path
    / "mobile_combustion/waterways_proxy.parquet"
):
    """
    Relative location information for waterways in the US.
    """

    waterways_loc = gpd.read_file(Waterways_file)

    waterways_loc = (
        waterways_loc.filter(items=['STATE', 'geometry'])
        .assign(geometry=lambda x: x['geometry'].apply(remove_z))
        .query("STATE not in ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI', 'XX', 'UM']")
        .drop(columns=['STATE'])
        .to_crs("ESRI:102003")
        .explode()
        .drop_duplicates(subset=['geometry'])
        )

    filtered_waterways = process_roads(waterways_loc)

    filtered_waterways = (
        filtered_waterways.explode()
        .drop_duplicates(subset=['geometry'])
        .to_crs("EPSG:4326")
        .dissolve(by=['STUSPS'])
        .reset_index()
        .rename(columns={'STUSPS': 'state_code',
                         'geometry': 'Geometry'})
        .set_geometry('Geometry')
    )

    # Save output
    filtered_waterways.to_parquet(reporting_proxy_output)
    return None


########################################################################################
# Building function
########################################################################################

# from shapely.geometry import LineString, MultiLineString

# def remove_z(geom):
#     if geom is None:
#         return None  # Handle None geometries
#     if geom.has_z:
#         # Handle LineString and MultiLineString separately
#         if isinstance(geom, LineString):
#             # Extract only X and Y from each point in the LineString
#             return LineString([(x, y) for x, y, z in geom.coords])
#         elif isinstance(geom, MultiLineString):
#             # Apply the same logic for each LineString in MultiLineString
#             return MultiLineString([LineString([(x, y) for x, y, z in line.coords]) for line in geom.geoms])
#     return geom


# # %% Read in Waterways spatial data
# waterways_loc = gpd.read_file(Waterways_file)

# waterways_loc = (
#     waterways_loc.filter(items=['STATE', 'geometry'])
#     .assign(geometry=lambda x: x['geometry'].apply(remove_z))
#     .query("STATE not in ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI', 'XX', 'UM']")
#     .drop(columns=['STATE'])
#     .to_crs("ESRI:102003")
#     .explode()
#     .drop_duplicates(subset=['geometry'])
#     )

# ####
# waterways_loc = waterways_loc[~waterways_loc['STATE'].isin(
#     ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI', 'XX', 'UM']
#     )]

# #waterways_loc = waterways_loc.explode()
# waterways_loc = waterways_loc.drop_duplicates(subset=['geometry'])
# filtered_waterways = filtered_waterways.drop_duplicates(subset=['geometry'])
# filtered_waterways = filtered_waterways.to_crs("EPSG:4326")


# ####



# # Read in State Spatial Data
# def read_states():
#     """
#     Read in State spatial data
#     """

#     gdf_states = gpd.read_file(gdf_state_files)

#     gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
#         ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
#         )]
#     gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]
#     gdf_states = gdf_states.to_crs("ESRI:102003")

#     return gdf_states

# states_gdf = read_states()

# import time
# import geopandas as gpd
# import pandas as pd
# from tqdm import tqdm

# def process_roads(waterways_data, states_gdf):
#     start_time = time.time()
#     final_roads = []

#     # Process each state
#     for _, state in tqdm(states_gdf.iterrows(), desc="Processing states"):
#         # Create single state GeoDataFrame
#         state_gdf = gpd.GeoDataFrame(geometry=[state.geometry], crs=states_gdf.crs)

#         # Clip roads to state boundary
#         state_roads = gpd.clip(waterways_data, state_gdf)

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

# filtered_waterways = process_roads(waterways_loc, states_gdf)

# filtered_waterways = filtered_waterways.explode()
# filtered_waterways = filtered_waterways.drop_duplicates(subset=['geometry'])
# filtered_waterways = filtered_waterways.to_crs("EPSG:4326")

# # Dissolve each GeoDataFrame by 'STUSPS' and 'year'
# dissolved_gdf = filtered_waterways.dissolve(by=['STUSPS'])
# # Reset the index
# dissolved_gdf = dissolved_gdf.reset_index()

# # Change column names
# waterways_proxy = dissolved_gdf.rename(columns={'STUSPS': 'state_code',
#                                                 'geometry': 'Geometry'})
# waterways_proxy = waterways_proxy.set_geometry('Geometry')


# #gdf.set_crs(epsg=4326, inplace=True)


# fig, ax = plt.subplots(figsize=(30, 20))

# gdf.plot(ax=ax,
#          markersize=gdf['SUM_Shape_Length'],
#          alpha=0.6,
#          cmap='viridis',
#          legend=True)

# plt.show()

# # %% Read in State spatial data
# gdf_states = gpd.read_file(gdf_state_files)

# gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
#     ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
#     )]
# gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]

# # %% Join Railroad and State spatial data

# gdf_lines = gdf
# gdf_lines = gdf_lines.set_crs(gdf_states.crs)

# waterway_gdf = gpd.sjoin(gdf_lines,
#                          gdf_states,
#                          how="left",
#                          predicate='within')

# waterway_gdf = waterway_gdf.dropna(subset=['STUSPS'])
# waterway_gdf = waterway_gdf[['STUSPS', 'geometry']]

# # %% Dissolve and make one GeoDataFrame
# dissolved_list = []


# # Dissolve each GeoDataFrame by 'STUSPS' and 'year'
# dissolved_gdf = waterway_gdf.dissolve(by=['STUSPS'])
# # Reset the index
# dissolved_gdf = dissolved_gdf.reset_index()

# # Change column names
# waterways_proxy = dissolved_gdf.rename(columns={'STUSPS': 'state_code',
#                                                 'geometry': 'Geometry'})
# waterways_proxy = waterways_proxy.set_geometry('Geometry')

# # %% Viz
# # Plot the geometry
# states_gdf_viz = states_gdf.to_crs("EPSG:4326")
# waterways_loc = waterways_loc.to_crs("EPSG:4326")

# fig, ax = plt.subplots(figsize=(30, 45))

# states_gdf_viz.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
# waterways_proxy.plot(ax=ax, color='blue', linewidth=1)
# #waterways_loc.plot(ax=ax, color='red', linewidth=1)


# plt.show()

# # %%

# water_test = gpd.read_file('/Users/aburnette/Downloads/ndc_761321034893104841.gpkg')

# testing = pd.read_csv('/Users/aburnette/Library/CloudStorage/OneDrive-SharedLibraries-EnvironmentalProtectionAgency(EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/emis/emi_waterways.csv')

# x_water_test = water_test[water_test['STATE'] == 'XX']
# y_water_test = water_test[water_test['STATE'] != 'XX']
# nc_water_test = water_test[water_test['STATE'] == 'NC']

# fig, ax = plt.subplots(figsize=(60, 40))

# #x_water_test.plot(ax=ax, color='blue', linewidth=1)
# #y_water_test.plot(ax=ax, color='red', linewidth=1)
# #  states_gdf_viz.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
# #waterways_loc.to_crs("EPSG:4326").plot(ax=ax, color='orange', linewidth=1)
# #waterways_test.to_crs("EPSG:4326").plot(ax=ax, color='green', linewidth=1)
# #coastline_2022.plot(ax=ax, color='green', linewidth=1)
# areawater_2022.plot(ax=ax, color='blue', linewidth=1)


# plt.show()



# ########################################################################################
# ########################################################################################
# # %% Exploring XX

# waterways_x= gpd.read_file(Waterways_file)

# waterways_x['geometry'] = waterways_x['geometry'].apply(remove_z)

# waterways_x = waterways_x.to_crs("ESRI:102003")

# waterways_x = waterways_x[waterways_x['STATE'] == 'XX']

# waterways_x = (
#     waterways_loc.filter(items=['STATE', 'geometry'])
#     .to_crs("ESRI:102003")
#     .explode()
#     )

# waterways_test = gpd.sjoin(waterways_x, states_gdf)

# ########################################################################################
# ########################################################################################
# ########################################################################################
# ########################################################################################
# # %% Testing new source
# coastline_2022 = gpd.read_file('/Users/aburnette/Downloads/tl_2022_us_coastline/tl_2022_us_coastline.shp')
# coastline_2022 = coastline_2022.to_crs("ESRI:102003")

# areawater_2022 = gpd.read_file('/Users/aburnette/Downloads/tl_2022_01001_areawater/tl_2022_01001_areawater.shp')
# areawater_2022 = areawater_2022.to_crs("ESRI:102003")

# linearwater_2022 = gpd.read_file('/Users/aburnette/Downloads/tl_2022_01001_linearwater/tl_2022_01001_linearwater.shp')
# linearwater_2022 = linearwater_2022.to_crs("ESRI:102003")
