from pathlib import Path


data_dir_path = Path("C:/Users/ccowell/Git/personal/GEPA/v3_data")
global_data_dir_path = data_dir_path / "global"
ghgi_data_dir_path = data_dir_path / "ghgi"
tmp_data_dir_path = data_dir_path / "tmp"


# this is used by the file task_download_census_geo.py to download specific census
# geometry files
census_geometry_list = ["primaryroads", "prisecroads", "rails"]

# List of reference boundaries (ex: state, county)
census_boundary_list = ["county", "state"]

# the years of the data processing.
min_year = 2012
max_year = 2022
years = range(min_year, max_year + 1)
