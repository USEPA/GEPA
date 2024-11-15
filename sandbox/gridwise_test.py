'''
This script uses duckdb to perform a spatial query on the roads dataset (which has been inserted as a table into a .duckdb file in script `roads_to_duckdb.py`).

The query being perfomed is at the output grid level (profile of which can be accessed using gch4i.gridding.GEPA_spatial_profile) and is a simple count of the number of roads in each grid cell.

Will likely want to reuse the get_vmt2 and get_vmt4 functions from the script `road_helpers.py`.
Those functions are used to read the VMT2 and VMT4 files, which are used to calculate the VMT for each road segment.

The output of this script will be a geodataframe with the grid cells and the length of roads in each cell-- or otherwise the emissions stats for each road type within each gridcell.

May also need to consider deduplication of roads within each grid cell. Another option there is to use the "reduce_roads" deduplication process run by Andrew in the original task_roads_proxy.py script (at least I think "reduce_roads" is the deduping process)
'''

#%%
# import duckdb
import pandas as pd
import geopandas as gpd
from shapely import from_wkt
from shapely.geometry import box

import rasterio as rio
import xarray as xr
import rioxarray as riox

from pathlib import Path
from datetime import datetime

from road_helpers import *

from gch4i.gridding import GEPA_spatial_profile
from gch4i.utils import get_cell_gdf

from gch4i.config import V3_DATA_PATH


def get_roads_path(year, raw_roads_path: Path=V3_DATA_PATH / "global/raw_roads"):
    return Path(raw_roads_path) / f"tl_{year}_us_allroads.parquet"

@benchmark_load
def read_roads(year, crs=4326):
    return gpd.read_parquet(get_roads_path(year)).to_crs(crs)

@benchmark_load
def intersect_sindex(cell, roads):
    '''
    Based of fthis geoff boeing blog post:
    https://geoffboeing.com/2016/10/r-tree-spatial-index-python/
    '''
    # first, add rtree spatial index to roads
    spatial_index = roads.sindex
    possible_matches_index = list(spatial_index.intersection(cell.bounds))
    possible_matches = roads.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(cell)]
    return precise_matches


if __name__ == '__main__':
    year = 2012
    road_proxy_df = get_road_proxy_data()
    cells = get_overlay_gdf(year)

    #%%
    roads = read_roads(year)

    for idx, cell in cells.iterrows():
        state = cell['state']
        region = cell['region']
        geom = box(cell['geometry'].bounds) # get box around cell for more efficient spatial query

        # get roads in cell
        cell_roads = intersect_sindex(geom, roads)





