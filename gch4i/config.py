from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)

# Set path to location of input data files (same directory where output files are saved)
# Define your own path to the data directory in the .env file
# Remove the environment variable
v3_data = os.getenv("V3_DATA_PATH")
V3_DATA_PATH = Path(v3_data)

figures_data_dir_path = V3_DATA_PATH / "figures"
global_data_dir_path = V3_DATA_PATH / "global"
ghgi_data_dir_path = V3_DATA_PATH / "ghgi"
tmp_data_dir_path = V3_DATA_PATH / "tmp"
emi_data_dir_path = V3_DATA_PATH / "emis"
proxy_data_dir_path = V3_DATA_PATH / "proxy"
sector_data_dir_path = V3_DATA_PATH / "sector"


# this is used by the file task_download_census_geo.py to download specific census
# geometry files
census_geometry_list = ["county", "state", "primaryroads"]

# the years of the data processing.
min_year = 2012
max_year = 2022
years = range(min_year, max_year + 1)

EQ_AREA_CRS = "ESRI:102003"



#######################
# data_load_functions
# Common GEPA functions to load global data files
#### Authors: 
# Erin E. McDuffie, Joannes D. Maasakkers, Candice F. Z. Chen
#### Date Last Updated: 
# Feb. 26, 2021

# Import modules
import pandas as pd
import numpy as np
# Load netCDF (for manipulating netCDF file types)
from netCDF4 import Dataset

# Define relative paths for common data files
def load_global_file_names():
    #National Data
    State_ANSI_inputfile = "../Global_InputData/ANSI/ANSI_state.txt" #0
    County_ANSI_inputfile = "../Global_InputData/ANSI/Counties_from_shape.csv" #1
    
    #Gridded input files
    pop_map_inputfile = '../Global_InputData/Gridded/census2010_population_c.nc' #2
    Grid_area01_inputfile = '../Global_InputData/Gridded/Gridded_area_c01.nc' #3
    Grid_area001_inputfile = '../Global_InputData/Gridded/Gridded_area_c.nc' #4
    Grid_state001_ansi_inputfile = '../Global_InputData/Gridded/State_id_001.nc' #5
    Grid_county001_ansi_inputfile= '../Global_InputData/Gridded/County_id_001.nc' #6
    
    return(State_ANSI_inputfile, County_ANSI_inputfile, pop_map_inputfile, \
          Grid_area01_inputfile, Grid_area001_inputfile, Grid_state001_ansi_inputfile, \
          Grid_county001_ansi_inputfile)

def load_road_globals():
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

    #  Define local variables
    start_year = min_year  # 2012 First year in emission timeseries
    end_year = max_year    # 2022 Last year in emission timeseries
    year_range = [*range(min_year, max_year+1, 1)]  # List of emission years
    year_range_str = [str(i) for i in year_range]
    num_years = len(year_range)

    return (
        raw_path, raw_roads_path, road_file, raw_road_file, task_outputs_path, global_path, 
        gdf_state_files, global_input_path, state_ansi_path, GEPA_Comb_Mob_path, State_vmt_file, 
        State_vdf_file, State_ANSI, name_dict, state_mapping, start_year, 
        end_year, year_range, year_range_str, num_years 
    )

#Load grid cell areas (m2), and lat and lon values for 0.01x0.01 grid
def load_area_map_001(Grid_area001_inputfile):
    #Read area map (0.01x0.01 degrees)
    area_file = Dataset(Grid_area001_inputfile)
    area_map001 = np.array(area_file.variables['cell_area'])
    lon001 = np.array(area_file.variables['lon'])
    lat001 = np.array(area_file.variables['lat'])
    area_file.close()
    #Get rid of missing values
    area_map001[area_map001 > 1.e+14] = 0
    
    return(area_map001, lat001, lon001)

#Load grid cell areas (m2), and lat and lon values for 0.1x0.1 grid
def load_area_map_01(Grid_area01_inputfile):
    #Read area map (0.1x0.1 degrees)
    area_file = Dataset(Grid_area01_inputfile)
    area_map01 = np.array(area_file.variables['cell_area'])
    lon01 = np.array(area_file.variables['LON'])
    lat01 = np.array(area_file.variables['LAT'])
    area_file.close()
    return(area_map01, lat01, lon01)

# Load Population Density map for 0.01x0.01 degree grid
def load_pop_den_map(pop_map_inputfile):
    pop_file = Dataset(pop_map_inputfile)
    pop_den_map001 = np.array(pop_file.variables['pop_density'])
    Lon = np.array(pop_file.variables['lon'])
    Lat = np.array(pop_file.variables['lat'])
    pop_file.close()
    #Set missing values to zero
    pop_den_map001[pop_den_map001 > 1] = 0.0
    return(pop_den_map001)

# Load State ANSI ID map for 0.01x0.01 degree grid
def load_state_ansi_map(Grid_state001_ansi_inputfile):
    state_file = Dataset(Grid_state001_ansi_inputfile)
    state_ANSI_map001 = np.array(state_file.variables['data'])
    state_file.close()
    return(state_ANSI_map001)

# Load County ANSI ID map for 0.01x0.01 degree grid
def load_county_ansi_map(Grid_county001_ansi_inputfile):
    county_file = Dataset(Grid_county001_ansi_inputfile)
    county_ANSI_map001 = np.array(county_file.variables['data'])
    county_file.close()
    return(county_ANSI_map001)
    
#Load the array of U.S. State ANSI IDs, names, and abbreviations
def load_state_ansi(state_ansi_inputfile):

    State_ANSI = pd.read_csv(state_ansi_inputfile, skiprows=1, sep="|", thousands=",", usecols=[0,1,2])
    State_ANSI.rename(columns={State_ANSI.columns[0]:'ansi'}, inplace=True)
    State_ANSI.rename(columns={State_ANSI.columns[1]:'abbr'}, inplace=True)
    State_ANSI.rename(columns={State_ANSI.columns[2]:'name'}, inplace=True)

    abbr_dict = State_ANSI.set_index('abbr')['ansi'].to_dict()
    name_dict = State_ANSI.set_index('name')['ansi'].to_dict()
    
    return(State_ANSI, name_dict, abbr_dict)


    