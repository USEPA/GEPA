from pathlib import Path


# data_dir_path = Path("C:/Users/ccowell/Environmental Protection Agency (EPA)/"
#                      "Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data")
data_dir_path = Path("C:/Users/ccowell/Git/personal/GEPA/v3_data")
global_data_dir_path = data_dir_path / "global"
ghgi_data_dir_path = data_dir_path / "ghgi"
tmp_data_dir_path = data_dir_path / "tmp"

# this is used by the file task_global_data_prep.py to download specific census
# geometry files
census_geometry_list = ["prisecroads", "rails", "uac"]

# this is used by the file task_global_data_prep.py to split specific census
# geometry files out by a specific field
split_geometry_dict = {"prisecroads": {"MTFCC": {"primroads": "S1100",
                                                 "secroads": "S1200"}}}
# this is used by the file task_global_data_prep.py to download specific census
# geometry files
grid_geometry_list = ["primroads", "secroads", "rails", "uac"]

# List of reference boundaries (ex: state, county)
census_boundary_list = ["county", "state"]

# the years of the data processing.
min_year = 2012
max_year = 2022
years = range(min_year, max_year + 1)
