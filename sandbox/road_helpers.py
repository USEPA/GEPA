########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
import os
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import numpy as np
import gc

from datetime import datetime

import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, GeometryCollection

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    max_year,
    min_year,
    years,
    load_state_ansi
)

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

########################################################################################
# %% STEP 0.2 Load Path Files
raw_path = Path(V3_DATA_PATH) / "global/raw"
raw_roads_path = Path(V3_DATA_PATH) / "global/raw_roads"

road_file = str(raw_path / "tl_")
raw_road_file = str(raw_roads_path / "tl_")

task_outputs_path = Path(V3_DATA_PATH) / "global/raw_roads/task_outputs"

global_path = Path(V3_DATA_PATH) / "global"
gdf_state_files = str(global_path / "tl_2020_us_state/tl_2020_us_state.shp")

global_input_path = Path(V3_DATA_PATH.parent) / "GEPA_Source_code/Global_InputData"
state_ansi_path = str(global_input_path / "ANSI/ANSI_state.txt")

GEPA_Comb_Mob_path = Path(V3_DATA_PATH.parent) / "GEPA_Source_Code/GEPA_Combustion_Mobile/InputData"
State_vmt_file = str(GEPA_Comb_Mob_path / "vm2/vm2_")
State_vdf_file = str(GEPA_Comb_Mob_path / "vm4/vm4_")

# ANSI State Codes
#State_ANSI, name_dict = data_load_fn.load_state_ansi(state_ansi_path)[0:2]
State_ANSI, name_dict = load_state_ansi(state_ansi_path)[0:2]

State_ANSI['State_Num'] = State_ANSI.index + 1
state_mapping = State_ANSI.set_index('State_Num')['abbr']

########################################################################################
# %% Define local variables
start_year = min_year  # 2012 First year in emission timeseries
end_year = max_year    # 2022 Last year in emission timeseries
year_range = [*range(min_year, max_year+1, 1)]  # List of emission years
year_range_str = [str(i) for i in year_range]
num_years = len(year_range)

# %% Functions
########################################################################################

def get_overlay_dir(year, 
                    out_dir: Path=task_outputs_path / 'overlay_cell_state_region'):
    return out_dir / f'cell_state_region_{year}.parquet'

def get_overlay_gdf(year):
    return gpd.read_parquet(get_overlay_dir(year))

# Read in State Spatial Data
def get_states_gdf(crs=4326):
    """
    Read in State spatial data
    """

    gdf_states = gpd.read_file(gdf_state_files)

    gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
        ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
        )]
    gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]
    gdf_states = gdf_states.to_crs(crs)

    return gdf_states

# Read in Region Spatial Data
def get_region_gdf(year, crs=4326):
    """
    Read in region spatial data
    """
    road_loc = (
        gpd.read_parquet(f"{road_file}{year}_us_uac.parquet", columns=['geometry'])
        .assign(year=year)
        .to_crs(crs)
        .assign(urban=1)
    )
    return road_loc

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


# Read in VM2 Data
def get_vm2_arrays(num_years):
    # Initialize arrays
    Miles_road_primary = np.zeros([2, len(State_ANSI), num_years])
    Miles_road_secondary = np.zeros([2, len(State_ANSI), num_years])
    Miles_road_other = np.zeros([2, len(State_ANSI), num_years])
    total = np.zeros(num_years)
    total2 = np.zeros(num_years)

    headers = ['STATE', 'RURAL - INTERSTATE', 'RURAL - FREEWAYS', 'RURAL - PRINCIPAL',
               'RURAL - MINOR', 'RURAL - MAJOR COLLECTOR', 'RURAL - MINOR COLLECTOR',
               'RURAL - LOCAL', 'RURAL - TOTAL', 'URBAN - INTERSTATE',
               'URBAN - FREEWAYS', 'URBAN - PRINCIPAL', 'URBAN - MINOR',
               'URBAN - MAJOR COLLECTOR', 'URBAN - MINOR COLLECTOR', 'URBAN - LOCAL',
               'URBAN - TOTAL', 'TOTAL']

    for iyear in np.arange(num_years):
        VMT_road = pd.read_excel(State_vmt_file + year_range_str[iyear] + '.xls',
                                 sheet_name='A',
                                 skiprows=13,
                                 nrows=51)
        VMT_road.columns = headers

        VMT_road = (VMT_road
                    .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                    .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                                   "District of Columbia" ""))
                    )

        for idx in np.arange(len(VMT_road)):
            VMT_road.loc[idx, 'ANSI'] = name_dict[VMT_road.loc[idx, 'STATE'].strip()]
            istate = np.where(VMT_road.loc[idx, 'ANSI'] == State_ANSI['ansi'])
            Miles_road_primary[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - INTERSTATE']
            Miles_road_primary[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - INTERSTATE']
            Miles_road_secondary[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - FREEWAYS'] + \
                VMT_road.loc[idx, 'URBAN - PRINCIPAL'] + \
                VMT_road.loc[idx, 'URBAN - MINOR']
            Miles_road_secondary[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - FREEWAYS'] + \
                VMT_road.loc[idx, 'RURAL - PRINCIPAL'] + \
                VMT_road.loc[idx, 'RURAL - MINOR']
            Miles_road_other[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - MAJOR COLLECTOR'] + \
                VMT_road.loc[idx, 'URBAN - MINOR COLLECTOR'] + \
                VMT_road.loc[idx, 'URBAN - LOCAL']
            Miles_road_other[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - MAJOR COLLECTOR'] + \
                VMT_road.loc[idx, 'RURAL - MINOR COLLECTOR'] + \
                VMT_road.loc[idx, 'RURAL - LOCAL']
            total[iyear] += np.sum(Miles_road_primary[:, istate, iyear]) + \
                np.sum(Miles_road_secondary[:, istate, iyear]) + \
                np.sum(Miles_road_other[:, istate, iyear])
            total2[iyear] += VMT_road.loc[idx, 'TOTAL']

        abs_diff = abs(total[iyear] - total2[iyear])/((total[iyear]+total2[iyear])/2)

        if abs(abs_diff) < 0.0001:
            print('Year ' + year_range_str[iyear] + ': Difference < 0.01%: PASS')
            print(total[iyear])
            print(total2[iyear])
        else:
            print('Year ' + year_range_str[iyear] + ': Difference > 0.01%: FAIL, diff: ' + str(abs_diff))
            print(total[iyear])
            print(total2[iyear])

    return Miles_road_primary, Miles_road_secondary, Miles_road_other, total, total2


# Read in VM4 Data
def get_vm4_arrays(num_years):
    # Initialize arrays
    Per_vmt_mot = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_pas = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_lig = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_hea = np.zeros([2, 3, len(State_ANSI), num_years])
    total_R = np.zeros(num_years)
    total_U = np.zeros(num_years)
    total = np.zeros(num_years)
    total2_U = np.zeros(num_years)
    total2 = np.zeros(num_years)
    total2_R = np.zeros(num_years)

    for iyear in np.arange(0, num_years):
        if year_range[iyear] == 2012 or year_range[iyear] == 2016:
            continue  # deal with missing data at the end
        else:
            # Read in Rural Sheet
            names = pd.read_excel(State_vdf_file + year_range_str[iyear]+'.xls',
                                  sheet_name='A', skiprows=12, header=0, nrows=1)
            colnames = names.columns.values
            VMT_type_R = pd.read_excel(State_vdf_file + year_range_str[iyear]+'.xls',
                                       na_values=['-'], sheet_name='A', names=colnames,
                                       skiprows=13, nrows=51)

            VMT_type_R.rename(columns={'MOTOR-': 'INTERSTATE - MOTORCYCLES',
                                       'PASSENGER': 'INTERSTATE - PASSENGER CARS',
                                       'LIGHT': 'INTERSTATE - LIGHT TRUCKS',
                                       'Unnamed: 4': 'INTERSTATE - BUSES',
                                       'SINGLE-UNIT': 'INTERSTATE - SINGLE-UNIT TRUCKS',
                                       'COMBINATION': 'INTERSTATE - COMBINATION TRUCKS',
                                       'Unnamed: 7': 'INTERSTATE - TOTAL',
                                       'MOTOR-.1': 'ARTERIALS - MOTORCYCLES',
                                       'PASSENGER.1': 'ARTERIALS - PASSENGER CARS',
                                       'LIGHT.1': 'ARTERIALS - LIGHT TRUCKS',
                                       'Unnamed: 11': 'ARTERIALS - BUSES',
                                       'SINGLE-UNIT.1': 'ARTERIALS - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.1': 'ARTERIALS - COMBINATION TRUCKS',
                                       'Unnamed: 14': 'ARTERIALS - TOTAL',
                                       'MOTOR-.2': 'OTHER - MOTORCYCLES',
                                       'PASSENGER.2': 'OTHER - PASSENGER CARS',
                                       'LIGHT.2': 'OTHER - LIGHT TRUCKS',
                                       'Unnamed: 18': 'OTHER - BUSES',
                                       'SINGLE-UNIT.2': 'OTHER - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.2': 'OTHER - COMBINATION TRUCKS',
                                       'Unnamed: 21': 'OTHER - TOTAL'}, inplace=True)

            VMT_type_R = (
                VMT_type_R
                .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                               "District of Columbia"))
                .assign(ANSI=0)
                .fillna(0)
                )

            # Read in Urban Sheet
            names = pd.read_excel(State_vdf_file + year_range_str[iyear] + '.xls',
                                  sheet_name='B', skiprows=12, header=0, nrows=1)
            colnames = names.columns.values
            VMT_type_U = pd.read_excel(State_vdf_file + year_range_str[iyear] + '.xls',
                                       na_values=['-'], sheet_name='B', names=colnames,
                                       skiprows=13, nrows=51)

            VMT_type_U.rename(columns={'MOTOR-': 'INTERSTATE - MOTORCYCLES',
                                       'PASSENGER': 'INTERSTATE - PASSENGER CARS',
                                       'LIGHT': 'INTERSTATE - LIGHT TRUCKS',
                                       'Unnamed: 4': 'INTERSTATE - BUSES',
                                       'SINGLE-UNIT': 'INTERSTATE - SINGLE-UNIT TRUCKS',
                                       'COMBINATION': 'INTERSTATE - COMBINATION TRUCKS',
                                       'Unnamed: 7': 'INTERSTATE - TOTAL',
                                       'MOTOR-.1': 'ARTERIALS - MOTORCYCLES',
                                       'PASSENGER.1': 'ARTERIALS - PASSENGER CARS',
                                       'LIGHT.1': 'ARTERIALS - LIGHT TRUCKS',
                                       'Unnamed: 11': 'ARTERIALS - BUSES',
                                       'SINGLE-UNIT.1': 'ARTERIALS - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.1': 'ARTERIALS - COMBINATION TRUCKS',
                                       'Unnamed: 14': 'ARTERIALS - TOTAL',
                                       'MOTOR-.2': 'OTHER - MOTORCYCLES',
                                       'PASSENGER.2': 'OTHER - PASSENGER CARS',
                                       'LIGHT.2': 'OTHER - LIGHT TRUCKS',
                                       'Unnamed: 18': 'OTHER - BUSES',
                                       'SINGLE-UNIT.2': 'OTHER - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.2': 'OTHER - COMBINATION TRUCKS',
                                       'Unnamed: 21': 'OTHER - TOTAL'}, inplace=True)

            VMT_type_U = (
                VMT_type_U
                .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                               "District of Columbia"))
                .assign(ANSI=0)
                .fillna(0)
                )

            # Distribute to 4 output types: passenger, light, heavy, motorcycle
            for idx in np.arange(len(VMT_type_R)):
                VMT_type_R.loc[idx, 'ANSI'] = name_dict[VMT_type_R.loc[idx, 'STATE']
                                                        .strip()]
                istate_R = np.where(VMT_type_R.loc[idx, 'ANSI'] == State_ANSI['ansi'])
                Per_vmt_mot[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - MOTORCYCLES']
                Per_vmt_mot[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - MOTORCYCLES']
                Per_vmt_mot[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - MOTORCYCLES']
                Per_vmt_pas[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - PASSENGER CARS']
                Per_vmt_pas[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - PASSENGER CARS']
                Per_vmt_pas[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - PASSENGER CARS']
                Per_vmt_lig[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - LIGHT TRUCKS']
                Per_vmt_lig[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - LIGHT TRUCKS']
                Per_vmt_lig[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - LIGHT TRUCKS']
                Per_vmt_hea[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - BUSES'] + \
                    VMT_type_R.loc[idx, 'INTERSTATE - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'INTERSTATE - COMBINATION TRUCKS']
                Per_vmt_hea[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - BUSES'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - COMBINATION TRUCKS']
                Per_vmt_hea[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - BUSES'] + \
                    VMT_type_R.loc[idx, 'OTHER - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'OTHER - COMBINATION TRUCKS']
                total_R[iyear] += np.sum(Per_vmt_mot[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_pas[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_lig[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_hea[1, :, istate_R, iyear])
                total2_R[iyear] += VMT_type_R.loc[idx, 'INTERSTATE - TOTAL'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - TOTAL'] + \
                    VMT_type_R.loc[idx, 'OTHER - TOTAL']

            for idx in np.arange(len(VMT_type_U)):
                VMT_type_U.loc[idx, 'ANSI'] = name_dict[VMT_type_U.loc[idx, 'STATE']
                                                        .strip()]
                istate_U = np.where(VMT_type_U.loc[idx, 'ANSI'] == State_ANSI['ansi'])
                Per_vmt_mot[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - MOTORCYCLES']
                Per_vmt_mot[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - MOTORCYCLES']
                Per_vmt_mot[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - MOTORCYCLES']
                Per_vmt_pas[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - PASSENGER CARS']
                Per_vmt_pas[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - PASSENGER CARS']
                Per_vmt_pas[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - PASSENGER CARS']
                Per_vmt_lig[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - LIGHT TRUCKS']
                Per_vmt_lig[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - LIGHT TRUCKS']
                Per_vmt_lig[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - LIGHT TRUCKS']
                Per_vmt_hea[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - BUSES'] + \
                    VMT_type_U.loc[idx, 'INTERSTATE - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'INTERSTATE - COMBINATION TRUCKS']
                Per_vmt_hea[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - BUSES'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - COMBINATION TRUCKS']
                Per_vmt_hea[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - BUSES'] + \
                    VMT_type_U.loc[idx, 'OTHER - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'OTHER - COMBINATION TRUCKS']
                total_U[iyear] += np.sum(Per_vmt_mot[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_pas[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_lig[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_hea[0, :, istate_U, iyear])
                total2_U[iyear] += VMT_type_U.loc[idx, 'INTERSTATE - TOTAL'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - TOTAL'] + \
                    VMT_type_U.loc[idx, 'OTHER - TOTAL']

            # Check for differences
            total[iyear] = total_U[iyear] + total_R[iyear]
            total2[iyear] = total2_R[iyear] + total2_U[iyear]
            abs_diff1 = abs(total[iyear] - total2[iyear]) / ((total[iyear] + total2[iyear]) / 2)

            if abs(abs_diff1) < 0.0001:
                print('Year ' + year_range_str[iyear] + ': Urban Difference < 0.01%: PASS')
            else:
                print('Year ' + year_range_str[iyear] + ': Urban Difference > 0.01%: FAIL, diff: ' + str(abs_diff1))
                print(total[iyear])
                print(total2[iyear])

    # Correct Years (assign 2012 to 2013), assign 2016 as average of 2015 and 2017
    idx_2012 = (2012-start_year)
    idx_2016 = (2016-start_year)
    Per_vmt_mot[:, :, :, idx_2012] = Per_vmt_mot[:, :, :, idx_2012 + 1]
    Per_vmt_pas[:, :, :, idx_2012] = Per_vmt_pas[:, :, :, idx_2012 + 1]
    Per_vmt_lig[:, :, :, idx_2012] = Per_vmt_lig[:, :, :, idx_2012 + 1]
    Per_vmt_hea[:, :, :, idx_2012] = Per_vmt_hea[:, :, :, idx_2012 + 1]

    Per_vmt_mot[:, :, :, idx_2016] = 0.5 * (Per_vmt_mot[:, :, :, idx_2016 - 1] +
                                            Per_vmt_mot[:, :, :, idx_2016 + 1])
    Per_vmt_pas[:, :, :, idx_2016] = 0.5 * (Per_vmt_pas[:, :, :, idx_2016 - 1] +
                                            Per_vmt_pas[:, :, :, idx_2016 + 1])
    Per_vmt_lig[:, :, :, idx_2016] = 0.5 * (Per_vmt_lig[:, :, :, idx_2016 - 1] +
                                            Per_vmt_lig[:, :, :, idx_2016 + 1])
    Per_vmt_hea[:, :, :, idx_2016] = 0.5 * (Per_vmt_hea[:, :, :, idx_2016 - 1] +
                                            Per_vmt_hea[:, :, :, idx_2016 + 1])

    # Optional: Combine Per_vmt_mot and Per_vmt_pas as Per_vmt_pas
    # Consult with EPA to determine whether to keep change
    Per_vmt_pas = Per_vmt_pas + Per_vmt_mot

    # Multiply by 0.01 to convert to percentage
    Per_vmt_mot = Per_vmt_mot * 0.01
    Per_vmt_pas = Per_vmt_pas * 0.01
    Per_vmt_lig = Per_vmt_lig * 0.01
    Per_vmt_hea = Per_vmt_hea * 0.01

    # Keep Per_vmt_mot reported for now, but it is now accounted for in Per_vmt_pas
    return Per_vmt_mot, Per_vmt_pas, Per_vmt_lig, Per_vmt_hea


# Calculate State Level Proxies
def calculate_state_proxies(num_years,
                            Miles_road_primary,
                            Miles_road_secondary,
                            Miles_road_other,
                            Per_vmt_pas,
                            Per_vmt_lig,
                            Per_vmt_hea):
    """
    array dimensions:
    region(urban/rural), road type(primary, secondary, other), state, year

    Example: vmt_pas : Miles_road_(primary, secondary, other) * Per_vmt_pas
    """

    # Initialize vmt_arrays
    vmt_pas = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_lig = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_hea = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_tot = np.zeros([2, len(State_ANSI), num_years])

    # Caclulate absolute number of VMT by region, road type, vehicle type, state, year
    # e.g. vmt_pas = VMT for passenger vehicles with dimensions = region (urban/rural),
    # road type (primary, secondary, other), state, and year

    # vmt_tot = region x state, year
    # road mile variable dimensions (urban/rural, state, year)

    for iyear in np.arange(0, num_years):
        vmt_pas[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_pas[:, 0, :, iyear]
        vmt_pas[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_pas[:, 1, :, iyear]
        vmt_pas[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_pas[:, 2, :, iyear]

        vmt_lig[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_lig[:, 0, :, iyear]
        vmt_lig[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_lig[:, 1, :, iyear]
        vmt_lig[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_lig[:, 2, :, iyear]

        vmt_hea[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_hea[:, 0, :, iyear]
        vmt_hea[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_hea[:, 1, :, iyear]
        vmt_hea[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_hea[:, 2, :, iyear]

        vmt_tot[:, :, iyear] += Miles_road_primary[:, :, iyear] + \
            Miles_road_secondary[:, :, iyear] + \
            Miles_road_other[:, :, iyear]

    # Initialize denominators
    tot_pas = np.zeros([len(State_ANSI), num_years])
    tot_lig = np.zeros([len(State_ANSI), num_years])
    tot_hea = np.zeros([len(State_ANSI), num_years])

    # Calculate total VMT for state/year by vehicle
    for istate in np.arange(0, len(name_dict)):
        for iyear in np.arange(0, num_years):
            tot_pas[istate, iyear] = np.sum(vmt_pas[:, :, istate, iyear])
            tot_lig[istate, iyear] = np.sum(vmt_lig[:, :, istate, iyear])
            tot_hea[istate, iyear] = np.sum(vmt_hea[:, :, istate, iyear])

    # Calculate proxy values
    pas_proxy = vmt_pas / tot_pas
    lig_proxy = vmt_lig / tot_lig
    hea_proxy = vmt_hea / tot_hea

    return pas_proxy, lig_proxy, hea_proxy, vmt_tot


# Unpack State Proxy Arrays
def unpack_state_proxy(state_proxy_array):
    reshaped_state_proxy = state_proxy_array.reshape(-1)

    row_index = np.repeat(['urban', 'rural'], 3 * 57 * num_years)
    col1_index = np.tile(np.repeat(['Primary', 'Secondary', 'Other'], 57 * num_years), 2)
    col2_index = np.tile(np.repeat(np.arange(1, 58), num_years), 2 * 3)
    col3_index = np.tile(np.arange(min_year, max_year + 1), 2 * 3 * 57)

    df = pd.DataFrame({
        'Region': row_index,
        'Road Type': col1_index,
        'State': col2_index,
        'Year': col3_index,
        'Proxy': reshaped_state_proxy
    })

    df['State_abbr'] = df['State'].map(state_mapping)

    cols = df.columns.tolist()
    state_index = cols.index('State')
    cols.insert(state_index + 1, cols.pop(cols.index('State_abbr')))
    df = df[cols]

    return df


# Unpack State Total Proxy Arrays
def unpack_state_allroads_proxy(vmt_tot):
    reshaped_state_proxy = vmt_tot.reshape(-1)

    row_index = np.repeat(['urban', 'rural'], 57 * num_years)
    col1_index = np.tile(np.repeat(np.arange(1, 58), num_years), 2)
    col2_index = np.tile(np.arange(min_year, max_year + 1), 2 * 57)

    df = pd.DataFrame({
        'Region': row_index,
        'State': col1_index,
        'Year': col2_index,
        'Proxy': reshaped_state_proxy
    })

    df['State_abbr'] = df['State'].map(state_mapping)

    cols = df.columns.tolist()
    state_index = cols.index('State')
    cols.insert(state_index + 1, cols.pop(cols.index('State_abbr')))
    df = df[cols]

    return df


# Generate Roads Proportions Data
def get_roads_proportion_data(pas_proxy, lig_proxy, hea_proxy):
    """
    Formats data for roads proxy emissions
    """
    # Add Vehicle Type column
    pas_proxy['Vehicle'] = 'Passenger'
    lig_proxy['Vehicle'] = 'Light'
    hea_proxy['Vehicle'] = 'Heavy'

    # Combine DataFrames
    vmt_roads_proxy = pd.concat([pas_proxy,
                                 lig_proxy,
                                 hea_proxy], axis=0).reset_index(drop=True)
    vmt_roads_proxy = (
        vmt_roads_proxy.rename(columns={'State_abbr': 'state_code',
                                        'Road Type': 'road_type'})
                       .rename(columns=lambda x: str(x).lower())
                       .query("state_code not in ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI', 'UM']")
    )
    vmt_roads_proxy = vmt_roads_proxy[['state_code', 'year', 'vehicle', 'region',
                                       'road_type', 'proxy']]

    vmt_roads_proxy.to_csv(task_outputs_path / "roads_proportion_data.csv", index=False)

    del pas_proxy, lig_proxy, hea_proxy

    gc.collect()

    return None


def get_road_proxy_data(road_proxy_out_path: Path=task_outputs_path / "roads_proportion_data.csv"):
    if not road_proxy_out_path.exists():
        # Proportional Allocation of Roads Emissions
        #################
        # VM2 Outputs
        Miles_road_primary, Miles_road_secondary, Miles_road_other, total, total2 = get_vm2_arrays(num_years)

        # VM4 Outputs
        Per_vmt_mot, Per_vmt_pas, Per_vmt_lig, Per_vmt_hea = get_vm2_arrays(num_years)

        # State Proxy Outputs
        pas_proxy, lig_proxy, hea_proxy, vmt_tot = calculate_state_proxies(num_years,
                                                                        Miles_road_primary,
                                                                        Miles_road_secondary,
                                                                        Miles_road_other,
                                                                        Per_vmt_pas,
                                                                        Per_vmt_lig,
                                                                        Per_vmt_hea)

        # Unpack State Proxy Outputs
        pas_proxy = unpack_state_proxy(pas_proxy)
        lig_proxy = unpack_state_proxy(lig_proxy)
        hea_proxy = unpack_state_proxy(hea_proxy)
        # tot_proxy = unpack_state_allroads_proxy(vmt_tot)

        # Generate Roads Proportions Data
        get_roads_proportion_data(pas_proxy, lig_proxy, hea_proxy)
    
    return pd.read_csv(road_proxy_out_path)
