"""
Name:                   ww_Centralized_emis.py
Date Last Modified:     2024-08-15
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of wastewater emis - mobile emissions to State, Year,
                        emissions format
Input Files:            - WW_State-level Estimates_90-22_27June2024.xlsx
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  - This version of emi mapping is draft for mapping .py files
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
import pandas as pd
from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    emi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import tg_to_kt

# %% STEP 1. Create Emi Mapping Functions

input_path = ghgi_data_dir_path / "wasterwater" / "WW_State-level Estimates_90-22_27June2024.xlsx"


emi_df = pd.read_excel(
    io=input_path,
    sheet_name="Domestic Emissions",
    skiprows=73,
    nrows=56,
    usecols="D:AK"
)

emi_df.columns.values[0] = "State"
emi_df.columns.values[1:] = list(range(1990, 2023))


emi_df = (
    emi_df.rename(columns=lambda x: str(x).lower())
    .set_index("state")
    .replace(0, pd.NA)
    .apply(pd.to_numeric, errors="coerce")
    .dropna(how="all")
    .fillna(0)
    .reset_index()
    .melt(id_vars="state", var_name="year", value_name="ghgi_ch4_kt")
    .astype({"year": int, "ghgi_ch4_kt": float})
    .fillna({"ch4_kt": 0})
    .query("year.between(@min_year, @max_year)")
    .groupby(["state", "year"])["ghgi_ch4_kt"]
    .sum()
    .reset_index()
)

output_path = ghgi_data_dir_path / "wasterwater" / "state_level_centralized_ww_treatment_emi.csv"

emi_df.to_csv(output_path, index=False)

# %%
