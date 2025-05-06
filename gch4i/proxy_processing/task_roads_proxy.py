'''
Name:                  task_roads_proxy.py
Date Last Modified:    2025-02-12
Purpose:               Generate population proxy data for emissions.
Input Files:           
    - download URL: 
Output Files:          
    - destination Path: 


THE FINAL OUTPUT FOR THIS SCRIPT CAN BE FOUND ON ONE DRIVE/SHARE POINT AT THE FOLLOWING PATH:
C:/Users/<USERNAME>/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/global/raw_roads/task_outputs/gridwise_outputs/*cell.csv

The files named gridwise_mapping_YYYY_cell.csv are the estimated methane output aggregated to the cell level.
    The variable for methane emission allocation ('methane_emission_allocation') represents the proportion of 
    state-wide methane emissions from that year that are attirbutable to that vehicle type/road/urbanicity(region) type in a given cell

The files named gridwise_mapping_YYYY.csv are the estimated methane ouput for each cell/state/region overlay by vehicle type and road type.
    This file is the fully disaggregated methane emission. The variable for methane emission in this dataset represents more of a "rate" that needs to be divided by m_road_in_state to achieve true estimate of methane emission ((long_table['rd_length'] / long_table['m_road_in_state']) * long_table['proxy'])

The files named gridwise_mapping_intermediate_YYYY.csv is intermediate output that is used to calculate the final output.
    This file is similar to the gridwise_mapping_YYYY.csv file, but it contains the raw proxy and road length for that overlay
'''


# %% STEP 0.1. Load Packages

from pathlib import Path
import os
from typing import Annotated
from pytask import Product, mark, task

from shapely import errors as se
from shapely import from_wkt

import pandas as pd
import numpy as np
import xarray as xr
import rasterio as rio
import gc

from datetime import datetime

import geopandas as gpd
import dask_geopandas as dgpd
from pyproj import CRS
from shapely.geometry import MultiLineString, LineString, GeometryCollection, box
from geocube.api.core import make_geocube
import rioxarray

import matplotlib.pyplot as plt

from gch4i.config import (
    V3_DATA_PATH,
    global_data_dir_path,
    proxy_data_dir_path,
    tmp_data_dir_path,
    max_year,
    min_year,
    years,
    load_state_ansi,
    load_road_globals
)

from gch4i.gridding import GEPA_spatial_profile
from gch4i.utils import *

"""
The global function path is commented out because it would work properly when running
in interactive terminal, but not when running pytask. I copied over the global functions
to my local config file during testing (quick/dirty fix for testing code).

See above: load_state_ansi from gch4i.config
"""
# import sys
# global_function_path = Path(V3_DATA_PATH.parent) / "GEPA_Source_Code/Global_Functions"
# sys.path.append(str(global_function_path))
# import data_load_functions as data_load_fn
# from data_load_functions import load_state_ansi


#%%

# Roads Proxy
@mark.persist
@task(id="roads_proxy")

def task_roads_proxy() -> None:

    #########################################################################
    # get gridded overlay and output to file
    out_dir = task_outputs_path / 'overlay_cell_state_region'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'RUNNING OVERLAY')
    print(f"NOTE: OVERLAY WILL TAKE ~2 HOURS PER YEAR TO RUN")
    for year in year_range:
        read_reduce_data(year)
        run_overlay_for_year(year, out_dir)


    #########################################################################
    # process the roads data

    urban_map = {1: 'urban', 0: 'rural'}
    task_outputs_path = task_outputs_path / "gridwise_outputs"
    task_outputs_path.mkdir(exist_ok=True, parents=True)


    for year in range(min_year, max_year+1):
        # if year < 2018:
        #     continue
        epsg = "ESRI:102003"
        road_type_col = 'road_type'
        road_proxy_df = get_road_proxy_data().rename(columns={'state_code': 'state'})


        cells = get_overlay_gdf(year, crs=epsg).rename(columns={'STUSPS': 'state', 'urban': 'region'})
        cells = cells.loc[~cells['state'].isna()]
        cells['region'] = cells['region'].map(urban_map)

        print(f'Reading roads')
        # first, check for reduced roads and run that if needed
        


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
        long_table = pd.read_csv(interm_out_path).dropna(subset='cell_id')



        print(f'\nGridwise processing for {year} complete: {datetime.now()}')

        print(f'Calculating emission allocatoin for {year}')

        # calculate road length by state:
        state_road_length = long_table.groupby(['state', 'year', 'road_type', 'region'])['rd_length'].sum().rename('m_road_in_state').reset_index()

        # join with long table:
        long_table = long_table.join(state_road_length.set_index(['state', 'year', 'road_type', 'region']), on=['state', 'year', 'road_type', 'region'])

        # calculate methane emission allocation at the cell/region level
        long_table['methane_emission_allocation'] = (long_table['rd_length'] / long_table['m_road_in_state']) * long_table['proxy']

        print(f'Saving to CSV... {datetime.now()}')
        long_table.to_csv(task_outputs_path / f"gridwise_mapping_{year}.csv", index=False)

        print(f'FULL PROCESSING FOR {year} COMPLETE: {datetime.now()}')

        del(roads)

        road_proxy_df = get_road_proxy_data().rename(columns={'state_code': 'state'})
        print(road_proxy_df.loc[road_proxy_df['year'] == year].groupby(['state', 'year', 'vehicle']).proxy.sum().reset_index())
        
        print(f'The long table should not sum to 1 bc the proxy value has been allocated to each urban/rural and road type in each cell')
        print(long_table.groupby(['vehicle', 'state', 'year']).proxy.sum())

        print(f"For example, here is one cell's allocation of proxies, which should sum to 1 if all 3 road_types are present in both urban/rural region")
        example = long_table.query("state=='WA' & year==2017 & vehicle=='Passenger' & cell_id==42072.0")
        print(example)
        print(example.proxy.sum())
        print(example.methane_emission_allocation.sum())
    
        # read state geometries from remote
        states_gdf = gpd.read_file(r'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip')
        ax = states_gdf.join(long_table.groupby(['state', 'year']).methane_emission_allocation.sum().reset_index().set_index('state'), on='STUSPS').plot(column='methane_emission_allocation', legend=True)
        ax.set_title(f'Methane Emission Allocation by State for {year}')
        print(long_table.groupby(by='state')['methane_emission_allocation'].sum().sort_values())
        print(state_road_length.sort_values(by='m_road_in_state'))

 
        print(f'AFTER HAVING DONE ALL OF THAT, EACH STATE SHOULD SUM TO 1.0')
        print(long_table.groupby(['state'])['methane_emission_allocation'].sum())


        # The result of the above output is a CSV file with the following columns:
        # - state
        # - region
        # - year
        # - road_type
        # - proxy
        # - rd_length
        # - cell_id
        # - m_road_in_state 
        # - methane_emission

        # NEXT AGGREGATING TO THE CELL/YEAR level
        # The variable for methane emission allocation represents the proportion of 
        # state-wide methane emissions from that year that are attirbutable to that vehicle type/road/urbanicity(region) type in a given cell
        cell_csv_path = task_outputs_path / f"gridwise_mapping_{year}_cell.csv"
        long_table.groupby(['cell_id'])['methane_emission_allocation'].sum()\
            .reset_index().to_csv(cell_csv_path, index=False)


    profile = GEPA_spatial_profile()
    out_profile = profile.profile.copy()
    cells = get_cell_gdf()
        
    # task_outputs_path = task_outputs_path / "gridwise_outputs"
    csv_path = task_outputs_path / f"gridwise_mapping_{year}.csv"
    long_table = pd.read_csv(csv_path).dropna(subset=['cell_id'])

    long_data = pd.concat([pd.read_csv(task_outputs_path / f"gridwise_mapping_{year}_cell.csv").groupby(['cell_id'])['methane_emission_allocation'].sum().rename(year) for year in years], axis=1)

    # convert proxy df to xarray/netcdf, with dimensions for x, y, and year
    proxy_out_path = V3_DATA_PATH / 'proxy'

    cells_proxy = cells.join(long_data, how='left').reset_index(names=['cell_id'])

    print(cells_proxy.notna().sum())

    # Extract latitude and longitude from the geometry column
    print(f'Getting centroids: {datetime.now()}')
    cells_proxy['y'] = cells_proxy['geometry'].apply(lambda geom: geom.exterior.coords.xy[1][0]) - 0.1
    cells_proxy['x'] = cells_proxy['geometry'].apply(lambda geom: geom.exterior.coords.xy[0][0]) + 0.1


    # sort by latitude first, then longitude
    # this will create a grid that is ordered by longitude, with the first row being the lowest latitude
    cells_proxy = cells_proxy.sort_values(['y', 'x'], ascending=[False, True])

    # extract the array for that year and reshape using the original profile's height and width
    # this will create an array with the same dimensions as the original grid
    proxy_array = cells_proxy[years].to_numpy().reshape(profile.profile['height'], profile.profile['width'], len(years)) #], order='F')
    lat_array = cells_proxy['y'].to_numpy().reshape(profile.profile['height'], profile.profile['width']) #], order='F')
    lon_array = cells_proxy['x'].to_numpy().reshape(profile.profile['height'], profile.profile['width']) #], order='F')


    plt.imshow(proxy_array[:, :, 0], cmap='viridis')

    # Create the xarray dataset
    ds = xr.Dataset(
        {
            'road_emissions': (['y', 'x', 'year'], proxy_array)
        },
        coords={
            'y': (['y'], np.unique(lat_array)[::-1]),
            'x': (['x'], np.unique(lon_array)),
            'year': (['year'], years),  # Assuming year_array has a third dimension for years
        }
    )

    ds['road_emissions'].sel(year=2018).plot()

    # reorder dimensions in xarray to ['year', 'y', 'x']
    ds = ds.transpose('year', 'y', 'x')

    input_path: Path = tmp_data_dir_path / "population_proxy_raw.tif"
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip"

    # read in the state file and filter to lower 48 + DC
    state_gdf = (
        gpd.read_file(state_geo_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    # read in the raw population data raster stack as a xarray dataset
    # masked will read the nodata value and set it to NA
    pop_ds = rioxarray.open_rasterio(input_path, masked=True).rename({"band": "year"})

    pop_ds["year"] = years

    with rio.open(input_path) as src:
        ras_crs = src.crs


    # read in roads_df by year
    long_table = pd.read_csv(csv_path)
    long_table['methane_emission_allocation'] = long_table['methane_emission_allocation'].fillna(0)
    state_grid = make_geocube(
        vector_data=state_gdf, measurements=["statefp"], like=pop_ds, fill=99
    )
    state_grid

    # Ensure that the coordinates match
    state_grid = state_grid.assign_coords({
        'x': ds.coords['x'],
        'y': ds.coords['y']
    })

    # assign the state grid as a new variable in the population dataset
    pop_ds["statefp"] = state_grid["statefp"]
    pop_ds


    # add statefp as a coordinate to the ds dataset
    ds = ds.assign_coords(statefp=state_grid['statefp'])

    # first plot the statefp as a check
    ds.statefp.plot()
    plt.show()

    # then plot a specific year's statefp to check that it's aligned acrossed years
    ds.isel(year=1).statefp.plot()
    plt.show()

    # define a function to normalize the population data by state and year
    def normalize(x):
        return x / x.sum()

    # apply the normalization function to the population data
    out_ds = (
        ds.groupby(["year", "statefp"])
        .apply(normalize)
        .sortby(["year", "y", "x"])
    )
    out_ds["road_emissions"].shape

    # check that the normalization worked
    all_eq_df = (
        out_ds["road_emissions"]
        .groupby(["statefp", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .assign(
            # NOTE: Due to floating point rouding, we need to check if the sum is
            # close to 1, not exactly 1.
            is_close=lambda df: (np.isclose(df["sum_check"], 1))
            | (np.isclose(df["sum_check"], 0))
        )
    )

    vals_are_one = all_eq_df["is_close"].all()
    print(all_eq_df)
    print(f"are all state/year norm sums equal to 1? {vals_are_one}")
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")

    # plot. Not hugely informative, but shows the data is there.
    out_ds["road_emissions"].sel(year=2020).plot.imshow()
    plt.show()


    output_path = proxy_data_dir_path / "roads_proxy.nc"
    out_ds["road_emissions"].transpose("year", "y", "x").round(10).rio.write_crs(
        ras_crs
    ).to_netcdf(output_path)

    out_ds["road_emissions"].round(10).rio.write_crs(ras_crs).to_netcdf(output_path.with_stem("road_proxy_noTranspose.nc"))

