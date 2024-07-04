"""
Name:                   coal_emis.py
Date Last Modified:     2024-07-03
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of coal emissions to State, Year, emissions format
Input Files:            - Coal_90-22_FRv1-InvDBcorrection.xlsx
Output Files:           - coal_post_surf_emi.csv, coal_post_under_emi.csv, coal_surf_emi.csv, coal_under_emi.csv
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
def get_coal_inv_data(input_path, output_path, subcategory):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - coal_post_surf
    - coal_post_under
    - coal_surf
    - coal_under
    """
    subcategory_strings = {
        "coal_post_surf": "Post-Mining (Surface)",
        "coal_post_under": "Post-Mining (Underground)",
        "coal_surf": "Surface Mining",
        "coal_under": "Liberated"
    }
    subcategory_string = subcategory_strings.get(subcategory)
    if subcategory_string is None:
        raise ValueError("Invalid subcategory. Please choose from coal_post_surf, coal_post_under, coal_surf, coal_under.")
    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=514,
            usecols="A:BA"
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # modify subcategory1 column
        .replace({"subcategory1": {"Underground Recovered & Used": "Underground Liberated"}})
        # get just methane emissions for the specified subcategory
        .query("(ghg == 'CH4') & (subcategory1.str.contains(@subcategory_string, regex=False))")
        # keep only columns that have georef or a year
        .filter(regex='georef|19|20')
        # set the index to state
        .rename(columns={"georef": "state_code"})
        .set_index("state_code")
        # convert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # correct column types
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # filter to required years
        .query("year.between(@min_year, @max_year)")
    )
    emi_df.to_csv(output_path, index=False)


# %% STEP 2. Set Input/Output Paths

# INPUT PATHS
inventory_workbook_path = ghgi_data_dir_path / "coal/Coal_90-22_FRv1-InvDBcorrection.xlsx"


# OUTPUT PATHS
output_path_coal_post_surf = V3_DATA_PATH / "emis/coal_post_surf_emi.csv"
output_path_coal_post_under = V3_DATA_PATH / "emis/coal_post_under_emi.csv"
output_path_coal_surf = V3_DATA_PATH / "emis/coal_surf_emi.csv"
output_path_coal_under = V3_DATA_PATH / "emis/coal_under_emi.csv"

# %% STEP 3. Function Calls

# Post-Mining (Surface)
get_coal_inv_data(inventory_workbook_path, output_path_coal_post_surf, "coal_post_surf")

# Post-Mining (Underground)
get_coal_inv_data(inventory_workbook_path, output_path_coal_post_under, "coal_post_under")

# Surface Mining
get_coal_inv_data(inventory_workbook_path, output_path_coal_surf, "coal_surf")

# Liberated | Recovered & Used
get_coal_inv_data(inventory_workbook_path, output_path_coal_under, "coal_under")
