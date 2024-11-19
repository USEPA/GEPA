'''
This script outputs an overlaid geoparquet file of grid, region, and state boundaries for each year of region data

Each year takes about 30 minutes to process. This could be sped up by parallelizing the processing of each year or using dask data frames
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
    print(f"Overlaying cell grid with state and region boundaries: {datetime.now()}")
    cell_state_region_gdf = gpd.overlay(cell_gdf, state_gdf, how='union')
    print(f"Overlayed cell grid with state boundaries: {datetime.now()}")
    cell_state_region_gdf = gpd.overlay(cell_state_region_gdf, region_gdf, how='union')
    
    # where urban is NaN, set to 0 (since the "region" dataset is urban geometries) and drop variable for year (since it is redundant and in the file name)
    cell_state_region_gdf['urban'] = cell_state_region_gdf['urban'].fillna(0).astype(int)
    cell_state_region_gdf.drop(columns=['year'], inplace=True)

    # drop rows with no cell_id, as this indicates that the geometry falls outside of the US and so doesn't need to be processed
    cell_state_region_gdf.dropna(subset=['cell_id'], inplace=True)

    return cell_state_region_gdf

@benchmark_load
def run_overlay_for_year(year, out_dir):
    # Save the overlaid geoparquet file
    out_path = get_overlay_dir(year, out_dir)
    if out_path.exists():
        print(f"File already exists for {year}: {out_path}")
        return None
    
    # Load the region, cell and state data
    print(f"Reading datasets for {year}: {datetime.now()}")
    cell_gdf = get_cell_gdf().to_crs(4326).reset_index().rename(columns={'index': 'cell_id'})
    region_gdf = get_region_gdf(year)
    state_gdf = get_states_gdf()

    # Overlay the cell grid with the state and region boundaries
    cell_state_region_gdf = overlay_cell_state_region(cell_gdf, region_gdf, state_gdf)

    cell_state_region_gdf.to_parquet(out_path)

    print(f"Saved overlaid geoparquet file for {year} to {out_dir / f'cell_state_region_{year}.parquet'}")



if __name__ == "__main__":
    out_dir = task_outputs_path / 'overlay_cell_state_region'
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in year_range:
        run_overlay_for_year(year, out_dir)

#%%
