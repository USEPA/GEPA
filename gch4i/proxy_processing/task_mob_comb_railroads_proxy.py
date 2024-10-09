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
    # max_year,
    # min_year
)

import geopandas as gpd

########################################################################################
# %% STEP 0.2 Load Path Files
raw_path = Path(V3_DATA_PATH) / "global/raw"
Railroad_file_a = str(raw_path / "tl_")

input_path = Path(V3_DATA_PATH).parent / "GEPA_Source_Code" / "GEPA_Combustion_Mobile" / "InputData"
gdf_state_files = str(input_path / "tl_2023_us_state/tl_2023_us_state.shp")

########################################################################################
# %% Define local variables
start_year = 2012  # First year in emission timeseries
end_year = 2022    # Last year in emission timeseries
year_range = [*range(start_year, end_year+1, 1)]  # List of emission years
year_range_str = [str(i) for i in year_range]
num_years = len(year_range)

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
    rail_parq_array = np.empty((num_years,), dtype=object)
    for iyear in np.arange(0, num_years):
        rail_loc = gpd.read_parquet(Railroad_file_a +
                                    year_range_str[iyear] +
                                    '_us_rails.parquet')
        rail_loc['year'] = year_range_str[iyear]
        rail_parq_array[iyear] = rail_loc

    # Read in State spatial data
    gdf_states = gpd.read_file(gdf_state_files)

    gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
        ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
        )]
    gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]

    # Join Railroad and State spatial data
    rail_state_array = np.empty((num_years,), dtype=object)
    for iyear in np.arange(0, num_years):
        gdf_lines = rail_parq_array[iyear]

        gdf_lines = gdf_lines.set_crs(gdf_states.crs)

        rail_state_array[iyear] = gpd.sjoin(gdf_lines,
                                            gdf_states,
                                            how="left",
                                            predicate='within')
        # Remove rows that are missing state designation
        rail_state_array[iyear] = rail_state_array[iyear].dropna(subset=['STUSPS'])
        # Keep only the columns of interest
        rail_state_array[iyear] = rail_state_array[iyear][
            ['year', 'STUSPS', 'geometry']
            ]

    # Disolve and make one GeoDataFrame
    dissolved_list = []

    for iyear in range(len(rail_state_array)):
        # Dissolve each GeoDataFrame by 'STUSPS' and 'year'
        dissolved_gdf = rail_state_array[iyear].dissolve(by=['STUSPS', 'year'])
        # Reset the index
        dissolved_gdf = dissolved_gdf.reset_index()
        # Append the dissolved GeoDataFrame to the list
        dissolved_list.append(dissolved_gdf)

    # Concatenate all dissolved GeoDataFrames into one large GeoDataFrame
    railroad_proxy = gpd.GeoDataFrame(pd.concat(dissolved_list, ignore_index=True))
    # Change column names
    railroad_proxy = railroad_proxy.rename(columns={'STUSPS': 'state_code',
                                                    'geometry': 'Geometry'})
    railroad_proxy = railroad_proxy.set_geometry('Geometry')

    # Save output
    railroad_proxy.to_parquet(reporting_proxy_output)
    return None


########################################################################################
# Building function
########################################################################################


# # %% Read in Railroad spatial data

# rail_parq_array = np.empty((num_years,), dtype=object)
# for iyear in np.arange(0, num_years):
#     rail_loc = gpd.read_parquet(Railroad_file_a +
    # year_range_str[iyear] + '_us_rails.parquet')

#     rail_loc['year'] = year_range_str[iyear]

#     rail_parq_array[iyear] = rail_loc

# ######################################################################################
# # %% Read in State spatial data
# gdf_states = gpd.read_file(gdf_state_files)

# gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
# ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
# )]
# gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]


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

# # Plot the geometry
# railroad_proxy = railroad_proxy.set_geometry('Geometry')

# gdf_filtered = railroad_proxy[railroad_proxy['year'] == '2012']

# fig, ax = plt.subplots(figsize=(10, 10))

# gdf_filtered.plot(ax=ax, color='blue', linewidth=1)

# plt.show()