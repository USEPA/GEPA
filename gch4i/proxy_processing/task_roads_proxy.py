########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
import os
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import numpy as np
import gc

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

#global_function_path = Path(V3_DATA_PATH.parent) / "GEPA_Source_Code/Global_Functions"
#os.chdir(global_function_path)
#import data_load_functions as data_load_fn
#from data_load_functions import load_state_ansi

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


# Read in VM2 Data
def read_vmt2(num_years):
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
def read_vmt4(num_years):
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


########################################################################################
# %% Proxy Functions

def read_reduce_data(year):
    """
    Read in All Roads data. Deduplicate and reduce data early.
    """
    # Order road types
    road_type_order = ['Primary', 'Secondary', 'Other']

    df = (gpd.read_parquet(f"{raw_road_file}{year}_us_allroads.parquet",
                           columns=['MTFCC', 'geometry'])
          .assign(year=year))

    road_data = (
        df.to_crs("ESRI:102003")
        .assign(
            geometry=lambda df: df.normalize(),
            road_type=lambda df: pd.Categorical(
                np.select(
                    [
                        df['MTFCC'] == 'S1100',
                        df['MTFCC'] == 'S1200',
                        df['MTFCC'].isin(['S1400', 'S1630', 'S1640'])
                    ],
                    [
                        'Primary',
                        'Secondary',
                        'Other'
                    ],
                    default=None
                ),
                categories=road_type_order,  # Define the categories
                ordered=True  # Ensure the categories are ordered
            )
        )
    )
    # Sort
    road_data = road_data.sort_values('road_type').reset_index(drop=True)
    # Explode to make LineStrings
    road_data = road_data.explode(index_parts=True).reset_index(drop=True)
    # Remove duplicates of geometries
    road_data = road_data.drop_duplicates(subset='geometry', keep='first')

    # Separate out Road Types
    prim_year = road_data[road_data['road_type'] == 'Primary']
    sec_year = road_data[road_data['road_type'] == 'Secondary']
    oth_year = road_data[road_data['road_type'] == 'Other']

    buffer_distance = 3  # meters

    # Set buffers
    prim_buffer = prim_year.buffer(buffer_distance)
    prim_buffer = gpd.GeoDataFrame(geometry=prim_buffer, crs=road_data.crs)

    prisec_buffer = pd.concat([prim_year, sec_year], ignore_index=True)
    prisec_buffer = prisec_buffer.buffer(buffer_distance)
    prisec_buffer = gpd.GeoDataFrame(geometry=prisec_buffer, crs=road_data.crs)

    # Overlay
    sec_red = gpd.overlay(sec_year, prim_buffer, how='difference')
    other_red = gpd.overlay(oth_year, prisec_buffer, how='difference')

    # Combine
    road_data = pd.concat([prim_year, sec_red, other_red], ignore_index=True)

    road_data = road_data[['year', 'road_type', 'geometry']]

    # Write to parquet
    road_data.to_parquet(task_outputs_path / f"reduced_roads_{year}.parquet")

    del road_data, prim_year, sec_year, oth_year, prim_buffer, prisec_buffer, sec_red, other_red

    gc.collect()

    return None


# Read in Region Spatial Data
def read_regions(year):
    """
    Read in region spatial data
    """
    road_loc = (
        gpd.read_parquet(f"{road_file}{year}_us_uac.parquet", columns=['geometry'])
        .assign(year=year)
        .to_crs("ESRI:102003")
    )
    return road_loc


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


# Extract Line geometries from Geometry Collections
def extract_lines(geom):
    if geom is None:
        return None
    if isinstance(geom, (LineString, MultiLineString)):
        # Return the geometry
        return geom
    elif isinstance(geom, GeometryCollection):
        # Filter the collection to include only LineString or MultiLineString, Remove Points
        lines = [g for g in geom.geoms if isinstance(g, (LineString, MultiLineString))]
        if len(lines) == 1:
            return lines[0]
        elif len(lines) > 1:
            return MultiLineString(lines)
    else:
        return None  # Only returns LineStrings and MultiLineStrings (no polygons or points)


# Process geometry column
def process_geometry_column(gdf):
    # Apply the extract_lines function
    gdf['geometry'] = gdf['geometry'].apply(extract_lines)

    # Remove rows where geometry is None
    gdf = gdf.dropna(subset=['geometry'])

    # Ensure the GeoDataFrame only contains LineString and MultiLineString
    gdf = gdf[gdf['geometry'].apply(lambda geom: isinstance(geom, (LineString, MultiLineString)))]

    # Explode MultiLineStrings into LineStrings
    gdf = gdf.explode(index_parts=True).reset_index(drop=True)

    return gdf


# Function to process each year's data
def process_year(year_data, year_region):
    # Ensure year_region only has necessary columns
    year_region = year_region[['geometry']]

    # Overlay for urban roads
    urban_roads = gpd.overlay(year_data, year_region, how='intersection',
                              keep_geom_type=False)
    urban_roads['region'] = 'urban'

    # Overlay for rural roads
    rural_roads = gpd.overlay(year_data, year_region, how='difference',
                              keep_geom_type=False)
    rural_roads['region'] = 'rural'

    # Combine results and select necessary columns
    combined = pd.concat([urban_roads, rural_roads])

    # Filter for LineStrings and MultiLineStrings
    combined = process_geometry_column(combined)

    return combined[['year', 'STUSPS', 'region', 'road_type', 'geometry']]


# Finishing function
def state_region_join(year):

    states_gdf = read_states()

    # Step 1: Filter to year
    road_data = gpd.read_parquet(task_outputs_path / f"reduced_roads_{year}.parquet")
    # Step 2: Overlay to cut at state boundaries
    road_data = gpd.overlay(road_data, states_gdf, how='identity', keep_geom_type=False)
    # Step 3: Remove geometries that re not LineStrings or MultiLineStrings               # Check here if an issue
    road_data = process_geometry_column(road_data)
    # Step 4: Use a state buffer to prevent accidental road reduction
    buffer_distance = 10  # meters
    states_gdf_buffer = states_gdf.buffer(buffer_distance)
    states_gdf_buffer = gpd.GeoDataFrame(geometry=states_gdf_buffer, crs=states_gdf.crs)
    states_gdf_buffer = states_gdf_buffer.join(states_gdf.drop(columns='geometry'))
    # Step 5 Spatial join to assign state attributes
    road_data = gpd.sjoin(road_data, states_gdf_buffer, how='left', predicate='within')
    # Step 6: Remove duplicate road segments (again, if new generated)
    road_data = road_data.drop_duplicates(subset=['geometry'])
    # Step 7: Clean table
    road_data = (
        road_data[['year', 'STUSPS_right', 'geometry', 'road_type']]
        .rename(columns={'STUSPS_right': 'STUSPS'})
        .dropna(subset=['STUSPS'])
        # .assign(year=year,
        #         type=road_type_label)
    )
    # Step 8: Garbage collection
    del states_gdf, states_gdf_buffer, buffer_distance

    gc.collect()

    # Part 3: Generate Roads by State, Region
    year_region = read_regions(year)
    result = process_year(road_data, year_region)

    # Combine results and dissolve
    result = result.to_crs(epsg=4326)
    result = result.dissolve(by=['year', 'STUSPS', 'region', 'road_type']).reset_index()

    result.to_parquet(task_outputs_path / f"final_roads_{year}.parquet")

    del road_data, year_region, result

    gc.collect()

    return None


##########
# %% Join Data Together
# Proportional and Proxy Data Join
def prop_proxy_join():
    """
    Join proportional and proxy data
    """

    # Read in Proportional Data
    proportional_proxy = pd.read_csv(task_outputs_path / "roads_proxy_proportions.csv")

    # Read in All Roads Data
    result_list = []
    for year in year_range:
        year_data = gpd.read_parquet(task_outputs_path / f"final_roads_{year}.parquet")
        result_list.append(year_data)
    all_roads_proxy = pd.concat(result_list, ignore_index=True)

    del result_list
    gc.collect()

    # Join proportional data with geometry data
    roads_proxy = pd.merge(
        proportional_proxy,
        all_roads_proxy,
        on=['state_code', 'year', 'region', 'road_type'],
        how='left'
    ).reset_index()

    # Convert back to a GeoDataFrame
    roads_proxy = gpd.GeoDataFrame(roads_proxy, geometry="geometry").to_crs(epsg=4326)

    return roads_proxy


########################################################################################
########################################################################################
########################################################################################
# %% Pytask

@mark.persist
@task(id="roads_proxy")
def task_get_roads_proxy(
    state_path: Path = gdf_state_files,
    reporting_proxy_output: Annotated[Path, Product] = proxy_data_dir_path
    / "mobile_combustion/roads_proxy.parquet",
):
    """
    Relative location information for raods in the US.
    """

    # Proportional Allocation of Roads Emissions
    ####################################################################################
    # VM2 Outputs
    Miles_road_primary, Miles_road_secondary, Miles_road_other, total, total2 = read_vmt2(num_years)

    # VM4 Outputs
    Per_vmt_mot, Per_vmt_pas, Per_vmt_lig, Per_vmt_hea = read_vmt4(num_years)

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

    # Roads Proxy Processing
    ####################################################################################

    # Read in All Roads data. Deduplicate and reduce data early.
    for year in year_range:
        read_reduce_data(year)

    """
    Current outputs ran through this point: read_reduce_data(year)
    * output reduced_roads_{year}.parquet & roads_proportion_data.csv to raw_roads>task_outputs folder

    Previous versions:
    * read in all years to one gdf and processed by year
        * can't hold all years in memory
        * Solution: read in one year at a time to process and write to disk
        * Doesn't need to be continuously read in and out, this was for testing steps

    Attempted Solutions:
    * DASK Geopandas : worked on one year at a time, took too long on region split
    * Parallel Processing : Ran into issues with batching (Geopandas CRS)
        * converted to WKT to batch and converted back to CRS when read in
        * briefly worked, kernel died, stopped experimentation to focus on other tasks
    * Sequential Processing (no parallel processing, just batched): 
        * Built out code, did not run yet

    File: processed_state_roads_2012.parquet
    * stored in task_outputs folder
    * took 2 hours to run
    * effectively deduplicated roads and clipped to state boundaries
        * Tested methods:
            * grouping by state and clipping at state boundaries (running for each state)
            was quicker than gpd.overlay( identity ) > add buffer to state polygons >
            sjoin( within )
    * Code was originally ran in a different script, will include below Pytask code

    read_reduce_data function
    * Initially, I read in roads and split by fips code to separate into states. This made
    it difficult to deduplicate roads, since roads that were duplicated across states
    would be deleted for one state and not the other. To avoid this, I read in all roads
    at the beginning, sorted by hierarchy (primary, secondary, other), processed/exploded
    (made all multilinestrings into linestrings), and deduplicated > then clipped by state
    polygons.
    * Code for State_Fips version exists in separate script (can include if necessary).

    """

    for year in year_range:
        state_region_join(year)

    # Join Proportional and Proxy Data
    ####################################################################################

    # Join Proportional and Proxy Data
    roads_proxy = prop_proxy_join()

    # Save output
    roads_proxy.to_parquet(reporting_proxy_output)
    return None

########################################################################################
########################################################################################
########################################################################################
########################################################################################
# %% Processed_state_roads method described above
states_gdf = read_states()
region_gdf = read_regions(2012)
roads_2012 = gpd.read_parquet(task_outputs_path / "reduced_roads_2012.parquet")

import time
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

def process_roads(road_data, states_gdf):
    start_time = time.time()
    final_roads = []

    # Process each state
    for _, state in tqdm(states_gdf.iterrows(), desc="Processing states"):
        # Create single state GeoDataFrame
        state_gdf = gpd.GeoDataFrame(geometry=[state.geometry], crs=states_gdf.crs)

        # Clip roads to state boundary
        state_roads = gpd.clip(road_data, state_gdf)

        if not state_roads.empty:
            # Add state identifier
            state_roads['STUSPS'] = state.STUSPS
        
        final_roads.append(state_roads)
        print(f"Processed {state.STUSPS}")

    # Combine all results
    final_result = pd.concat(final_roads)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")

    return final_result

processed_roads_2012 = process_roads(roads_2012, states_gdf)
# 2 hr 50 min

# Make all LineStrings
processed_roads_2012 = process_geometry_column(processed_roads_2012)

# Write out
#processed_roads_2012.to_parquet(task_outputs_path / "processed_state_roads_2012.parquet")
processed_roads_2012 = gpd.read_parquet(task_outputs_path / "processed_state_roads_2012.parquet")

########################################################################################
########################################################################################
# %% Dask (commented out sections during testing) and Not Dask: Previous Method for state joins

import dask_geopandas as dgpd

"""
Dask code is messy because I ran into issues with deduplication, and attempted to
troubleshoot everything in Dask.
In hindsight, I should have just used my reduce method applied above and attempted
dask on the region split and/or the state join.
"""

year_range = [2020, 2021, 2022]


def read_reduce_data(year):
    """
    Read in All Roads data. Deduplicate and reduce data early.
    """
    road_data = (gpd.read_parquet(f"{raw_road_file}{year}_us_allroads.parquet",
                                  columns=['MTFCC', 'geometry'])
                 .assign(year=year))

    print(type(road_data))

    road_data = dgpd.from_geopandas(road_data, npartitions=4)

    print("Type after Dask transformation:", type(road_data))
    print("Columns:", road_data.columns)

    # Order road types
    road_type_order = ['Primary', 'Secondary', 'Other']

    # Define road type
    # def assign_road_type(df):
    #     df['road_type'] = (np.select(
    #         [
    #             df['MTFCC'] == 'S1100',
    #             df['MTFCC'] == 'S1200',
    #             df['MTFCC'].isin(['S1400', 'S1630', 'S1640'])
    #         ],
    #         [
    #             'Primary',
    #             'Secondary',
    #             'Other'
    #         ],
    #         default=None
    #     )
    #     return df

    road_data = (
        road_data.map_partitions(
            lambda df: df.assign(
                road_type=pd.Categorical(
                    np.select(
                        [
                            df['MTFCC'] == 'S1100',
                            df['MTFCC'] == 'S1200',
                            df['MTFCC'].isin(['S1400', 'S1630', 'S1640'])
                        ],
                        [
                            'Primary',
                            'Secondary',
                            'Other'
                        ],
                        default=None
                    ),
                    categories=road_type_order,
                    ordered=True
                )
            ),
            meta={'MTFCC': 'object',
                  'geometry': 'geometry',
                  'year': 'int64',
                  'road_type': 'category'}
        )
    )

    print("Columns after road_type_creation:", road_data.columns)
    print("Type after road_type_creation:", type(road_data))

    # Convert back to Dask GeoDataFrame
    def dask_to_dgdf(dask_df):
        return dgpd.from_dask_dataframe(dask_df, geometry='geometry')

    # Convert back
    road_data = dask_to_dgdf(road_data)
    road_data = road_data.set_crs("ESRI:102003", allow_override=True)

    #road_data = road_data[['year', 'road_type', 'geometry']]
    #print("Columns after column reduction:", road_data.columns)

    # Order road types
    # road_type_order = ['Primary', 'Secondary', 'Other']

    #road_data['road_type'] = pd.Categorical(road_data['road_type'], categories=road_type_order, ordered=True)
    #road_data = road_data.sort_values('road_type').reset_index(drop=True)
    def sort_by_road_type(road_data):
        road_data_sorted = road_data.map_partitions(
            lambda df: df.sort_values('road_type').reset_index(drop=True),
            meta={'MTFCC': 'object',
                  'geometry': 'geometry',
                  'year': 'int64',
                  'road_type': 'category'}
        )
        return road_data_sorted

    road_data = sort_by_road_type(road_data)

    print("Columns after sort:", road_data.columns)

    print("Before explode:", type(road_data))
    # Convert back
    road_data = dask_to_dgdf(road_data)
    road_data = road_data.set_crs("ESRI:102003", allow_override=True)
    print("After convert back:", type(road_data))

    # Explode to make LineStrings
    road_data = road_data.map_partitions(
        lambda df: gpd.GeoDataFrame(df).explode(column='geometry')
    ).reset_index(drop=True)

    print("After explode:", type(road_data))
    road_data = dask_to_dgdf(road_data)
    road_data = road_data.set_crs("ESRI:102003", allow_override=True)
    print("After convert back:", type(road_data))
    # Remove duplicates of geometries
    road_data = road_data.drop_duplicates(subset='geometry')

    print("After drop_duplicates:", type(road_data))
    road_data = dask_to_dgdf(road_data)
    road_data = road_data.set_crs("ESRI:102003", allow_override=True)

    # Compute
    road_data = road_data.compute()

    print("After compute", type(road_data))

    if not isinstance(road_data, gpd.GeoDataFrame):
        road_data = gpd.GeoDataFrame(road_data, crs="ESRI:102003")

    print(type(road_data))

    # Adjust columns
    road_data = road_data[['year', 'road_type', 'geometry']]

    road_data = dgpd.from_geopandas(road_data, npartitions=4)

    print("type after transform:", type(road_data))

    # Separate out Road Types
    prim_year = road_data[road_data['road_type'] == 'Primary']
    sec_year = road_data[road_data['road_type'] == 'Secondary']
    oth_year = road_data[road_data['road_type'] == 'Other']

    buffer_distance = 3  # meters

    meta = {
        'year': 'int64',
        'road_type': 'category',
        'geometry': 'geometry'
    }

    # Define buffer_geometries
    def buffer_geometries(gdf, buffer_distance):
        # Buffer geometries
        gdf['geometry'] = gdf.geometry.buffer(buffer_distance)
        return gdf

    def apply_overlay(target_gdf, buffer_gdf):
        return gpd.overlay(target_gdf.compute(), buffer_gdf.compute(), how='difference')

    def process_year_data(prim_year, sec_year, oth_year, buffer_distance):
        # Set buffers
        prim_year_buffer = prim_year.map_partitions(buffer_geometries, buffer_distance, meta=meta)
        prisec_year = dd.concat([prim_year, sec_year], axis=0)
        prisec_buffer = prisec_year.map_partitions(buffer_geometries, buffer_distance, meta=meta)

        # Apply overlay
        sec_red = apply_overlay(sec_year, prim_year_buffer)
        other_red = apply_overlay(oth_year, prisec_buffer)

        return prim_year, sec_red, other_red

    prim_year, sec_red, other_red = process_year_data(prim_year, sec_year, oth_year, buffer_distance)

    # Combine
    road_data = dd.concat([prim_year, sec_red, other_red], axis=0)

    road_data = road_data[['year', 'road_type', 'geometry']]

    # Compute
    road_data = road_data.compute()

    road_data.to_parquet(task_outputs_path / f"reduced_roads_{year}.parquet")

    del road_data, prim_year, sec_year, oth_year

    gc.collect()

    return None


######
# my_road_data = read_reduce_data()
for year in year_range:
    read_reduce_data(year)
    print(f"Reduced roads for {year}")
######


########################################################################################
# %% Non Dask Version

year_range = [2020, 2021]


def read_reduce_data_nodask(year):
    """
    Read in All Roads data. Deduplicate and reduce data early.
    """
    # Order road types
    road_type_order = ['Primary', 'Secondary', 'Other']

    df = (gpd.read_parquet(f"{raw_road_file}{year}_us_allroads.parquet",
                           columns=['MTFCC', 'geometry'])
          .assign(year=year))

    road_data = (
        df.to_crs("ESRI:102003")
        .assign(
            geometry=lambda df: df.normalize(),
            road_type=lambda df: pd.Categorical(
                np.select(
                    [
                        df['MTFCC'] == 'S1100',
                        df['MTFCC'] == 'S1200',
                        df['MTFCC'].isin(['S1400', 'S1630', 'S1640'])
                    ],
                    [
                        'Primary',
                        'Secondary',
                        'Other'
                    ],
                    default=None
                ),
                categories=road_type_order,  # Define the categories
                ordered=True  # Ensure the categories are ordered
            )
        )
    )
    # Sort
    road_data = road_data.sort_values('road_type').reset_index(drop=True)
    # Explode to make LineStrings
    road_data = road_data.explode(index_parts=True).reset_index(drop=True)
    # Remove duplicates of geometries
    road_data = road_data.drop_duplicates(subset='geometry', keep='first')

    # Separate out Road Types
    prim_year = road_data[road_data['road_type'] == 'Primary']
    sec_year = road_data[road_data['road_type'] == 'Secondary']
    oth_year = road_data[road_data['road_type'] == 'Other']

    buffer_distance = 3  # meters

    # Set buffers
    prim_buffer = prim_year.buffer(buffer_distance)
    prim_buffer = gpd.GeoDataFrame(geometry=prim_buffer, crs=road_data.crs)

    prisec_buffer = pd.concat([prim_year, sec_year], ignore_index=True)
    prisec_buffer = prisec_buffer.buffer(buffer_distance)
    prisec_buffer = gpd.GeoDataFrame(geometry=prisec_buffer, crs=road_data.crs)

    # Overlay
    sec_red = gpd.overlay(sec_year, prim_buffer, how='difference')
    other_red = gpd.overlay(oth_year, prisec_buffer, how='difference')

    # Combine
    road_data = pd.concat([prim_year, sec_red, other_red], ignore_index=True)

    road_data = road_data[['year', 'road_type', 'geometry']]

    # Write to parquet
    road_data.to_parquet(task_outputs_path / f"reduced_roads_{year}.parquet")

    del road_data, prim_year, sec_year, oth_year, prim_buffer, prisec_buffer, sec_red, other_red

    gc.collect()

    return None

########################################################################################


for year in year_range:
    read_reduce_data_nodask(year)
    print(f"Reduced roads for {year}")