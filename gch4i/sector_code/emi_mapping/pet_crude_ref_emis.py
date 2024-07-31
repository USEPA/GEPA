"""
Name:                   pet_crude_ref_emis.py
Date Last Modified:     2024-07-24
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of petroleum (Crude Oil Refining - Total) - mobile
                        emissions to State, Year, emissions format
Input Files:            InvDB_Petroleum_Systems_StateData_2024GHGI.xlsx
                        State_Summary_gaskt tab]
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

# %% STEP 1. Create Emi Mapping Functions


def get_pet_crude_ref_inv_data(input_path, output_path):

    emi_df = pd.read_excel(
        inventory_workbook_path,
        sheet_name="State_Summary_gaskt",
        skiprows=2,
        nrows=575,
        index_col=None
        )

    emi_df = emi_df.dropna(how="all", axis=1)

    state_list = emi_df["State"].unique()
    # Remove non-continental states
    state_list = [state for state in state_list if state not in
                  ["AS", "GU", "MP", "PR", "VI"]]

    emi_df = emi_df.rename(columns=lambda x: str(x).lower()) \
                   .query('ghg == "CH4" and subref == "Crude Oil Refining"') \
                   .drop(columns={"subref", "ghg"}) \
                   .query("`state` in @state_list") \
                   .set_index("state") \
                   .apply(pd.to_numeric, errors="coerce") \
                   .replace(0, pd.NA) \
                   .dropna(how="all") \
                   .fillna(0) \
                   .reset_index() \
                   .melt(id_vars="state", var_name="year", value_name="ch4_kt") \
                   .astype({"year": int, "ch4_kt": float}) \
                   .fillna({"ch4_kt": 0}) \
                   .query("year.between(@min_year, @max_year)")
    emi_df.to_csv(output_path, index=False)


# %% STEP 2. Set Input/Output Paths

# INPUT PATHS
inventory_workbook_path = ghgi_data_dir_path / "Petroleum and Natural Gas/InvDB_Petroleum_Systems_StateData_2024GHGI.xlsx"

# OUTPUT PATHS
output_path = tmp_data_dir_path / "pet_crude_ref_emi.csv"

# %% STEP 3. Function Calls

get_pet_crude_ref_inv_data(
    inventory_workbook_path,
    output_path
    )
