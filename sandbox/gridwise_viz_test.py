'''
This script uses duckdb to perform a spatial query on the roads dataset (which has been inserted as a table into a .duckdb file in script `roads_to_duckdb.py`).

The query being perfomed is at the output grid level (profile of which can be accessed using gch4i.gridding.GEPA_spatial_profile) and is a simple count of the number of roads in each grid cell.

Will likely want to reuse the get_vmt2 and get_vmt4 functions from the script `road_helpers.py`.
Those functions are used to read the VMT2 and VMT4 files, which are used to calculate the VMT for each road segment.

The output of this script will be a geodataframe with the grid cells and the length of roads in each cell-- or otherwise the emissions stats for each road type within each gridcell.

May also need to consider deduplication of roads within each grid cell. Another option there is to use the "reduce_roads" deduplication process run by Andrew in the original task_roads_proxy.py script (at least I think "reduce_roads" is the deduping process)
'''

#%%
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import from_wkt

import rasterio as rio
import xarray as xr
import rioxarray as riox

from pathlib import Path
from datetime import datetime

from gch4i.gridding import GEPA_spatial_profile
from gch4i.utils import get_cell_gdf

from gch4i.config import V3_DATA_PATH


def benchmark_load(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {datetime.now() - start}")
        return result
    return wrapper

def get_roads_path(year, raw_roads_path: Path=V3_DATA_PATH / "global/raw_roads"):
    return Path(raw_roads_path) / f"tl_{year}_us_allroads.parquet"

@benchmark_load
def read_roads(year):
    return gpd.read_parquet(get_roads_path(year)).to_crs("ESRI:4326")


# Define local variables
raw_path = Path(V3_DATA_PATH) / "global/raw"
raw_roads_path = Path(V3_DATA_PATH) / "global/raw_roads"
# task_outputs_path = Path(V3_DATA_PATH) / "global/raw_roads/task_outputs"

# duck db file
duckdb_path = Path('data/roads.duckdb')
table_prefix = 'roads'
duckdb_path.parent.mkdir(parents=True, exist_ok=True)

def get_urban_cell():
    # cells = get_cell_gdf()
    # urban, LA cell
    # cell = cells.loc[145716, 'geometry']
    # print(cell)
    cell = from_wkt("POLYGON ((-118.4 34.2, -118.4 34.099999999999994, -118.3 34.099999999999994, -118.3 34.2, -118.4 34.2))")
    return cell

def get_rural_cell():
    # cells = get_cell_gdf()
    # rural Montana cell
    # cell = cells.loc[44980, 'geometry']
    # print(cell)
    cell = from_wkt("POLYGON ((-112 48.6, -112 48.5, -111.9 48.5, -111.9 48.6, -112 48.6))")
    return cell


#%%

# cells = get_cell_gdf()
cell = get_urban_cell()
print(cell)
year = 2012
roads = read_roads(year)


#%%
@benchmark_load
def intersect_sindex(cell, roads):
    # first, add rtree spatial index to roads
    spatial_index = roads.sindex
    possible_matches_index = list(spatial_index.intersection(cell.bounds))
    possible_matches = roads.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(cell)]
    return precise_matches

result = intersect_sindex(cell, roads)

#%%
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


#%%
# clip the roads to the cell bounds and plot again

clipped = result.copy()
clipped['geometry'] = clipped.intersection(cell)
clipped = clipped[clipped.geometry.is_empty == False]

fig, ax = plt.subplots()
gpd.GeoSeries(cell).plot(ax=ax, color='red')
clipped.plot(ax=ax, color='blue')
plt.show()




