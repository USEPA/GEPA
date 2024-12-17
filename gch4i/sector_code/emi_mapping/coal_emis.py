"""
Name:                   coal_emis.py
Date Last Modified:     2024-07-02
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of coal emissions to State, Year, emissions format
Input Files:            - Coal_90-22_FRv1-InvDBcorrection.xlsx
Output Files:           - Do not use output. This is a single function script.
Notes:                  - This version of emi mapping is draft for mapping .py files
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
import pandas as pd
from gch4i.utils import tg_to_kt


from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)


# %% STEP 1. Create Emi Mapping Functions


def get_underground_liberated_coal_inv_data(input_path, output_path):
    """read in the ch4_kt values for each state"""

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
        # get just methane emissions
        .query("(ghg == 'CH4') & (subcategory1.str.contains('Liberated'))")
        # drop columns we don't need (Leave GeoRef, [Years])
        .drop(
            columns=[
                "sector",
                "category",
                "subcategory1",
                "subcategory2",
                "subcategory3",
                "subcategory4",
                "subcategory5",
                "carbon pool",
                "fuel1",
                "fuel2",
                "exclude",
                "id",
                "sensitive (y or n)",
                "data type",
                "subsector",
                "crt code",
                "units",
                "ghg",
                "gwp"
            ]
        )
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


# %% Input and Output File paths
# INVENTORY INPUT FILE
inventory_workbook_path = ghgi_data_dir_path / "coal/Coal_90-22_FRv1-InvDBcorrection.xlsx"

# INVENTORY OUTPUT FILE
output_path_liberated = V3_DATA_PATH / "emis/under_lib_emi.csv"

# %% Use Function
get_underground_liberated_coal_inv_data(inventory_workbook_path, output_path_liberated)
