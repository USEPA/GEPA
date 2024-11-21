'''
This script uses duckdb to perform a spatial query on the roads dataset (which has been inserted as a table into a .duckdb file in script `roads_to_duckdb.py`).

The query being perfomed is at the output grid level (profile of which can be accessed using gch4i.gridding.GEPA_spatial_profile) and is a simple count of the number of roads in each grid cell.

Will likely want to reuse the get_vmt2 and get_vmt4 functions from the script `road_helpers.py`.
Those functions are used to read the VMT2 and VMT4 files, which are used to calculate the VMT for each road segment.

The output of this script will be a geodataframe with the grid cells and the length of roads in each cell-- or otherwise the emissions stats for each road type within each gridcell.

May also need to consider deduplication of roads within each grid cell. Another option there is to use the "reduce_roads" deduplication process run by Andrew in the original task_roads_proxy.py script (at least I think "reduce_roads" is the deduping process)
'''

#%%
%reload_ext autoreload
%autoreload 2
# import duckdb
import pandas as pd
import geopandas as gpd
from shapely import from_wkt
from shapely.geometry import box
from shapely.ops import unary_union
from shapely import errors as se

import rasterio as rio
import xarray as xr
import rioxarray as riox

from pathlib import Path
from datetime import datetime

from road_helpers import *

from gch4i.gridding import GEPA_spatial_profile
from gch4i.utils import get_cell_gdf

from gch4i.config import V3_DATA_PATH

# from gridwise_viz_test import get_urban_cell

#%%

def get_urban_cell():
    # cells = get_cell_gdf()
    # urban, LA cell
    # cell = cells.loc[145716, 'geometry']
    # print(cell)
    cell = from_wkt("POLYGON ((-118.4 34.2, -118.4 34.099999999999994, -118.3 34.099999999999994, -118.3 34.2, -118.4 34.2))")
    return cell

def plot_result(cell, result):
    # plot the cell and the resulting intersected roads
    # plot a grid of 2 plots
    # in plot 1, plot the full extent
    # in plot 2, set max extent to the cell bounds
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    gpd.GeoSeries(cell).plot(ax=ax[0], color='red')
    result.plot(ax=ax[0], color='blue')
    ax[0].set_title("Full extent")
    ax[1].set_title("Cell bounds")
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    gpd.GeoSeries(cell).plot(ax=ax[1], color='red')
    result.plot(ax=ax[1], color='blue')
    ax[0].set_title("Full extent")
    ax[1].set_title("Cell bounds")
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    # set the extent of the plot to the cell bounds
    ax[1].set_xlim(cell.bounds[0], cell.bounds[2])
    ax[1].set_ylim(cell.bounds[1], cell.bounds[3])

    plt.show()

urban_map = {1: 'urban', 0: 'rural'}
task_outputs_path = Path('tmp/gridwise_test_outputs')
task_outputs_path.mkdir(exist_ok=True, parents=True)

#%%

if __name__ == '__main__':
    for year in range(min_year, max_year+1):
        if year < 2018:
            continue
        epsg = "ESRI:102003"
        road_type_col = 'road_type'
        road_proxy_df = get_road_proxy_data().rename(columns={'state_code': 'state'})
        cells = get_overlay_gdf(year, crs=epsg).rename(columns={'STUSPS': 'state', 'urban': 'region'})
        cells = cells.loc[~cells['state'].isna()]
        cells['region'] = cells['region'].map(urban_map)

        print(f'Reading roads')
        roads = read_roads(year, crs=epsg, raw=False)

        print(f"Starting processing for {year} at {datetime.now()}")
        interm_out_path = task_outputs_path / f"gridwise_mapping_intermediate_{year}.csv"

        if not interm_out_path.exists():
            # create an empty CSV file to store intermediate results
            pd.DataFrame(columns=['state', 'region', 'year', 'road_type', 'cell_id', 'vehicle', 'proxy', 'rd_length']).to_csv(interm_out_path, index=False)
        
        else:
            # read in the intermediate file and filter cells that have already been processed based on ['cell_id', 'state', 'region']
            processed_cells = pd.read_csv(interm_out_path)
            cells = cells.loc[~cells.set_index(['cell_id', 'state', 'region']).index.isin(processed_cells.set_index(['cell_id', 'state', 'region']).index)]
            # remove the intermediate file from environment
            del(processed_cells)

        cell_n = len(cells)
        for idx, cell in cells.reset_index().iterrows():
            print(f'Processing index {idx}/{cell_n}', end='\r')
            cell_id = cell['cell_id']
            state = cell['state']
            region = cell['region']
            geom = cell['geometry'] # get box around cell for more efficient spatial query

            # get road value from proxy data
            road_value = road_proxy_df.loc[(road_proxy_df['state'] == state) & (road_proxy_df['region'] == region) & (road_proxy_df['year'] == year)]

            # get roads in cell
            try:
                cell_roads = intersect_and_clip(roads, geom)

            except se.GEOSException as e:
                geom = geom.buffer(0)

                try:
                    cell_roads = intersect_and_clip(roads, geom)
            
                except Exception as e:
                    print(f'ERROR processing cell {cell_id}, region: {region}, state: {state}, for year {year}: {e}')
                    continue
            # Calculate the length of lines within each road_type group
            # first check if DF has values. intersect_and_clip returns an empty DF if no roads are found
            if not cell_roads.empty:
                road_type_df = (
                    cell_roads
                    .groupby(road_type_col, observed=True)['geometry']
                    .apply(lambda x: x.length.sum()).rename('rd_length')
                    .to_frame().reset_index()
                    .assign(state=state, region=region, year=year, cell_id=cell_id)
                    .set_index(['state', 'region', 'year', 'cell_id', road_type_col])
                )
            else:
                # if no roads in cell, create a DF where rd_length are 0 and join
                road_type_df = pd.DataFrame({road_type_col: roads[road_type_col].unique(), 'rd_length': 0}).assign(state=state, region=region, year=year, cell_id=cell_id, rd_length=0).set_index(['state', 'region', 'year', 'cell_id', road_type_col])

            # merge with road_value and write to CSV
            road_value.set_index(['state', 'region', 'year', 'road_type']).join(road_type_df, how='left').reset_index()\
                .to_csv(interm_out_path, mode='a', header=False, index=False)

        # concatenate
        long_table = pd.read_csv(interm_out_path)

        print(f'\nGridwise processing for {year} complete: {datetime.now()}')

        print(f'Calculating emission allocatoin for {year}')
        # calculate road length by state:
        state_road_length = long_table.groupby(['state', 'year'])['rd_length'].sum().rename('m_road_in_state').reset_index()

        # join with long table:
        long_table = long_table.join(state_road_length.set_index(['state', 'year']), on=['state', 'year'])

        # calculate methane emission allocation at the cell/region level
        long_table['methane_emission_allocation'] = (long_table['rd_length'] / long_table['m_road_in_state']) * long_table['proxy']

        print(f'Saving to CSV... {datetime.now()}')
        long_table.to_csv(task_outputs_path / f"gridwise_mapping_{year}.csv", index=False)

        print(f'FULL PROCESSING FOR {year} COMPLETE: {datetime.now()}')

        del(roads)



        # The result of this script is a CSV file with the following columns:
        # - state
        # - region
        # - year
        # - road_type
        # - proxy
        # - rd_length
        # - cell_id
        # - m_road_in_state 
        # - methane_emission

        # The variable for methane emission allocation represents the proportion of 
        # state-wide methane emissions from that year that are attirbutable to that vehicle type/road/urbanicity(region) type in a given cell
                
        # There's a chance that these will need to be re-aggregated to the cell level as follows:
        long_table.groupby(['year', 'cell_id'])['methane_emission_allocation'].sum()\
            .reset_index().to_csv(task_outputs_path / f"gridwise_mapping_{year}_cell.csv", index=False)

#%%
# WORKING THEORY:
# I think i need to obtain the proportion of road length in a cell to the entire state's length of that road type in order to calculate the miles travelled by a given car type on that road type in that cell

# get length of road type in that cell/state/region_year
road_length = cell.loc[cell['road_type'] == 'Primary'].length.sum()

# get emissions for each vehicle type in each year in each state
for idx, group_df in road_proxy_df.loc[road_proxy_df['state_code'] == cell.state.unique()[0]].groupby(['vehicle', 'year']):
    # multiply road length by emissions per mile from the road_proxy_df dataset
    # this is something that may not have been calculated yet
    # may need to look up the v2 code to see how they got emissions per mile for each vehilce type


    break





