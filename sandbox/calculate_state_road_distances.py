'''
This script loads the reduced_roads dataset, converts to an equal-length projection, calculates cumulative length of each road type in each state, and saves the result to a csv

Output is a csv crosswalk with columns for state, road type, and cumulative length of that road type in that state
'''

#%%
%reload_ext autoreload
%autoreload 2
import geopandas as gpd
import pandas as pd
import dask_geopandas as dgpd
from shapely.geometry import box

import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from gch4i.config import (
    V3_DATA_PATH,
    min_year,
    max_year
)
from road_helpers import read_roads, get_states_gdf, benchmark_load, intersect_and_clip
#%%
@benchmark_load
def overlay_roads_on_states(roads, states, dask=False):
    if not dask:
        # Overlay the roads with the state boundaries
        print(f"Overlaying roads with state boundaries: {datetime.now()}")
        roads = gpd.overlay(roads, states, how='intersection')
        print(f"Overlay complete: {datetime.now()}")
        return roads
    else:
        print(f'USING DASK')
        # converting to dask:
        # convert roads to a dask geopandas dataframe
        roads = dgpd.from_geopandas(roads, npartitions=4)
        # convert states to a dask geopandas dataframe
        states = dgpd.from_geopandas(states, npartitions=4)

        # Overlay the roads with the state boundaries
        print(f"Overlaying roads with state boundaries: {datetime.now()}")
        result = dgpd.overlay(roads, states, how='intersection')
        print(f"Overlay complete: {datetime.now()}")
        return result.compute()
    


#%%
crs = "ESRI:102003" # Albers USA Contiguous Equidistant Conic
state_name_col = 'STUSPS'
road_type_col = 'road_type'

if __name__ == '__main__':
    # for year in range(min_year, max_year+1):
    crosswalk_list = []
    for year in [2012]:
        # Read the roads data
        roads = read_roads(year, raw=False, crs=crs)  # raw=False means we are reading the reduced roads dataset
        # Get the state boundaries
        states = get_states_gdf(crs=crs)

        #%%
        
        for idx, state in states.iterrows():
            state_abbrev = state[state_name_col]
            print(f"Getting roads that intersect with {state_abbrev}")
            state_geom = state['geometry']
            state_roads = intersect_and_clip(roads, state_geom, state_abbrev, simplify=10000, dask=True)

            if not state_roads.empty:
                for road_type, subset in state_roads.groupby(road_type_col):
                    print(f"Calculating length of {road_type} roads in {state_abbrev}")
                    crosswalk_list.append({
                        'state': state_abbrev,
                        'year': year,
                        'road_type': road_type,
                        'length': subset.length.sum()
                    })
            print()

        #%%)
        
        # Save the result to a geoparquet file
        out_path = Path(V3_DATA_PATH) / f"global/raw_roads/task_outputs/state_road_lengths_{year}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(crosswalk_list).to_parquet(out_path)

        print(f"Saved state road lengths for {year} to {out_path}")

#%%