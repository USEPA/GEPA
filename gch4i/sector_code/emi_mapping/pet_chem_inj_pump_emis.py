"""
Name:                   pet_checm_inj_pump_emis.py
Date Last Modified:     2024-07-25
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of petroleum chemical injection pump - mobile emissions
                        to State, Year, emissions format
Input Files:            - ChemicalInjectionPumps_StateEstimates_2024.xlsx
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  - This version of emi mapping is draft for mapping .py files
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
import pandas as pd
from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import tg_to_kt

# %% STEP 1. Create Emi Mapping Functions

def get_pet_chem_inj_pump_inv_data(input_path, output_path):
    """
    This function will read in the CH4 emissions from Petroleum emissions and
    generate state emissions table with state, yeear, and CH4 emissions in kt.
    """
    emi_df = pd.read_excel(
        inventory_workbook_path,
        sheet_name="Petro_State_CH4_MT",
        nrows=57,
        index_col=None
        )

    emi_df = emi_df.dropna(how="all", axis=1)

    state_list = emi_df["State Code"].unique()
    # Remove non-continental states
    state_list = [state for state in state_list if state not in
                  ["AS", "GU", "MP", "PR", "VI"]]

    emi_df = emi_df.rename(columns=lambda x: str(x).lower()) \
                   .drop(columns="state") \
                   .query("`state code` in @state_list") \
                   .set_index("state code") \
                   .apply(pd.to_numeric, errors="coerce") \
                   .replace(0, pd.NA) \
                   .dropna(how="all") \
                   .fillna(0) \
                   .reset_index() \
                   .melt(id_vars="state code", var_name="year", value_name="ch4_tg") \
                   .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt) \
                   .drop(columns=["ch4_tg"]) \
                   .astype({"year": int, "ch4_kt": float}) \
                   .fillna({"ch4_kt": 0}) \
                   .query("year.between(@min_year, @max_year)")
    emi_df.to_csv(output_path, index=False)


# %% STEP 2. Set Input/Output Paths

# INPUT PATHS
inventory_workbook_path = ghgi_data_dir_path / "Petroleum and Natural Gas/ChemicalInjectionPumps_StateEstimates_2024.xlsx"

# OUTPUT PATHS
output_path = tmp_data_dir_path / "pet_chem_inj_pump_emi.csv"

# %% STEP 3. Function Calls

testing = get_pet_chem_inj_pump_inv_data(
    inventory_workbook_path,
    output_path
    )
