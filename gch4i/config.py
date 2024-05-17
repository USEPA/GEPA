from pathlib import Path

# Set path to location of input data files (same directory where output files are saved)
data_dir_path = Path(
    "~/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data"
).expanduser()
# EEM path
# data_dir_path = Path("C:/Users/EMCDUF01/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Documents/RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data")

global_data_dir_path = data_dir_path / "global"
ghgi_data_dir_path = data_dir_path / "ghgi"
tmp_data_dir_path = data_dir_path / "tmp"


# this is used by the file task_download_census_geo.py to download specific census
# geometry files
census_geometry_list = ["county", "state", "primaryroads"]

# the years of the data processing.
min_year = 2012
max_year = 2022
years = range(min_year, max_year + 1)
