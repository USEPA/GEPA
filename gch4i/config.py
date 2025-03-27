from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Set path to location of input data files (same directory where output files are saved)
# Define your own path to the data directory in the .env file
v3_data = os.getenv("V3_DATA_PATH")
V3_DATA_PATH = Path(v3_data)

figures_data_dir_path = V3_DATA_PATH / "figures"
global_data_dir_path = V3_DATA_PATH / "global"
ghgi_data_dir_path = V3_DATA_PATH / "ghgi"
tmp_data_dir_path = V3_DATA_PATH / "tmp"
emi_data_dir_path = V3_DATA_PATH / "emis"
proxy_data_dir_path = V3_DATA_PATH / "proxy"
sector_data_dir_path = V3_DATA_PATH / "sector"
intermediate_data_dir_path = V3_DATA_PATH / "interim"

prelim_gridded_dir = V3_DATA_PATH / "gridded_data_prelim"
final_gridded_dir = V3_DATA_PATH / "gridded_data_final"

logging_dir = V3_DATA_PATH.parents[0] / "gridding_log_and_qc"
gridded_output_dir = logging_dir / "gridded_output"

# this is used by the file task_download_census_geo.py to download specific census
# geometry files
census_geometry_list = ["county", "state", "primaryroads"]

# the years of the data processing.
min_year = 2012
max_year = 2022
years = range(min_year, max_year + 1)

EQ_AREA_CRS = "ESRI:102003"
