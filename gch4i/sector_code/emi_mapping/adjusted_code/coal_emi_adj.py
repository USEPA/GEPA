"""
Name:                   coal_emi_adj.py
Date Last Modified:     2024-08-13
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of coal emissions to State, Year, emissions format
Input Files:            - Coal_90-22_FRv1-InvDBcorrection.xlsx
Output Files:           - coal_post_surf_emi.csv, coal_post_under_emi.csv,
                        coal_surf_emi.csv, coal_under_emi.csv
Notes:                  - This version of emi mapping is draft for mapping .py files
"""
# %% STEP 0. Load packages, configuration files, and local parameters ------------------
import pandas as pd
import ast
from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import tg_to_kt

from gch4i.sector_code.emi_mapping.a_excel_dict import (read_excel_dict, read_excel_dict2, read_excel_dict_cell, file_path)

# %% STEP 0.5 Establish Paths
source_name = "coal_mining"

proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_data = pd.read_excel(proxy_file_path, sheet_name="testing").query(
    f"gch4i_name == '{source_name}'"
)

emi_parameters_dict = {}

for emi_name, data in proxy_data.groupby("emi"):
    emi_parameters_dict[emi_name] = {
        "input_path": ghgi_data_dir_path / source_name / data.file_name.iloc[0],
        "source_list": data.ghgi_group.to_list(),
        "output_path": tmp_data_dir_path / f"{emi_name}.csv",
    }


# %% STEP 1. Create Emi Mapping Functions
