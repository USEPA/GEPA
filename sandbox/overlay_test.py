'''
This script outputs an overlaid geoparquet file of grid, region, and state boundaries for each year of region data
'''

#%%
import geopandas as gpd
import pandas as pd
import numpy as np

from datetime import datetime

from road_helpers import *
from gch4i.config import V3_DATA_PATH

from gch4i.gridding import GEPA_spatial_profile
from gch4i.utils import get_cell_gdf

import gch4i.config as config

@benchmark_load
def overlay_cell_state_region(cell_gdf, region_gdf, state_gdf):
    '''
    This function overlays the cell grid with the state and region boundaries for each year of region data
    '''    
    # Overlay the cell grid with the state and region boundaries
    cell_state_region_gdf = gpd.overlay(cell_gdf, state_gdf, how='intersection')
    cell_state_region_gdf = gpd.overlay(cell_state_region_gdf, region_gdf, how='intersection')
    
    return cell_state_region_gdf


#%%
year=2012
# Load the region, cell and state data
print(f"Reading datasets for {year}: {datetime.now()}")
cell_gdf = get_cell_gdf().to_crs(4326).reset_index().rename(columns={'index': 'cell_id'})
region_gdf = get_region_gdf(year)
state_gdf = get_states_gdf()


#%%
cell = gpd.GeoDataFrame(cell_gdf.loc[110952].to_frame().T, crs=4326)
state = state_gdf.loc[state_gdf['NAME'].isin(['Missouri', 'Kansas'])]
region = gpd.GeoDataFrame(region_gdf.loc[1574].to_frame().T, crs=4326).assign(urban=1)

#%%
# plot all three layers together in a single plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
state.plot(ax=ax, color='red')
region.plot(ax=ax, color='blue')
cell.plot(ax=ax, color='white')

# set extent to region
ax.set_xlim(region.total_bounds[[0, 2]])
ax.set_ylim(region.total_bounds[[1, 3]])

plt.show()

#%%

@benchmark_load
def overlay_cell_state_region(cell_gdf, region_gdf, state_gdf):
    '''
    This function overlays the cell grid with the state and region boundaries for each year of region data
    '''    
    # Overlay the cell grid with the state and region boundaries
    cell_state_region_gdf = gpd.overlay(cell_gdf, state_gdf, how='union')
    cell_state_region_gdf = gpd.overlay(cell_state_region_gdf, region_gdf, how='union')
    
    # where urban is NaN, set to 0 (since the "region" dataset is urban geometries) and drop variable for year (since it is redundant and in the file name)
    cell_state_region_gdf['urban'] = cell_state_region_gdf['urban'].fillna(0).astype(int)
    cell_state_region_gdf.drop(columns=['year'], inplace=True)

    # drop rows with no cell_id, as this indicates that the geometry falls outside of the US and so doesn't need to be processed
    cell_state_region_gdf.dropna(subset=['cell_id'], inplace=True)

    return cell_state_region_gdf

result = overlay_cell_state_region(cell, region, state)

fig, ax = plt.subplots()
result.plot(ax=ax, column='cell_id')
# set extent to region
ax.set_xlim(cell.total_bounds[[0, 2]])
ax.set_ylim(cell.total_bounds[[1, 3]])
plt.show()

#%%
for idx, row in result.iterrows():
    fig, ax = plt.subplots()
    gpd.GeoSeries(row['geometry']).plot(ax=ax, color='blue')
    # set extent to region
    ax.set_xlim(region.total_bounds[[0, 2]])
    ax.set_ylim(region.total_bounds[[1, 3]])
    plt.show()
# Overlay the cell grid with the state and region boundaries
# cell_state_region_gdf = overlay_cell_state_region(year, cell_gdf, region_gdf, state_gdf)
