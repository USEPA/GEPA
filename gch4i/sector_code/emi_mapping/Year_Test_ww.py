"""
Name:                   ww_emis.py
Date Last Modified:     2024-07-19
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


def get_ww_inv_data(input_path, output_path, subcategory):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - ww_ad_emi
    - ww_centralaerobic_emi
    - ww_centralanaerobic_emi
    - ww_septic_emi
    - ww_brew_emi
    - ww_ethanol_emi
    - ww_fv_emi
    - ww_mp_emi
    - ww_petrref_emi
    - ww_pp_emi
    """

    # Table Dimensions
    # pp B16 - AI72
    # mp D84 - AL139
    # fv E138 - AL194
    # petrref B19 - AI75
    # ethanol B13 - AI69
    # brew B17 - AI73

    subcategory_strings = {
        "ww_brew_emi": ["Breweries Emissions", 16, 56, "B:AI"],
        "ww_ethanol_emi": ["Ethanol Emissions", 12, 56, "B:AI"],
        "ww_petrref_emi": ["Petroleum Emissions", 18, 56, "B:AI"],
        "ww_fv_emi": ["F_V_J Emissions", 137, 56, "E:AL"],
        "ww_mp_emi": ["M_P Emissions", 82, 56, "D:AL"],  # No column Headers
        "ww_pp_emi": ["Pulp and Paper Emissions", 15, 56, "B:AI"]
    }
    subcategory_string = subcategory_strings.get(subcategory)
    if subcategory_string is None:
        raise ValueError("""Invalid subcategory. Please choose from ww_ad_emi,
                        ww_centralaerobic_emi, ww_centralanaerobic_emi, ww_septic_emi,
                        ww_brew_emi, ww_ethanol_emi, ww_fv_emi, ww_mp_emi,
                        ww_petrref_emi, ww_pp_emi.""")

    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name=subcategory_string[0],
            skiprows=subcategory_string[1],
            nrows=subcategory_string[2],
            usecols=subcategory_string[3]
        )
    )
    # Will need an if-else statement for ww_mp_emi. Adjust for no column headers
    if subcategory == "ww_mp_emi":
        emi_df.columns.values[0] = "State"
        emi_df.columns.values[1] = "Metric"
        emi_df.columns.values[2:] = list(range(1990, 2023))
        emi_df = emi_df.drop(columns="Metric")

    state_list = emi_df["State"].unique()
    # Remove non-continental states
    state_list = [state for state in state_list if state not in ["AS", "GU", "MP", "PR", "VI", "AL", "HI"]] # Add National (state code)

    emi_df = emi_df.rename(columns=lambda x: str(x).lower()) \
                   .query("`state` in @state_list") \
                   .set_index("state") \
                   .apply(pd.to_numeric, errors="coerce") \
                   .dropna(how="all") \
                   .replace(0, pd.NA) \
                   .dropna(how="all") \
                   .reset_index() \
                   .melt(id_vars="state", var_name="year", value_name="ch4_tg") \
                   .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt) \
                   .drop(columns=["ch4_tg"]) \
                   .astype({"year": int, "ch4_kt": float}) \
                   .fillna({"ch4_kt": 0}) \
                   .query("year.between(@min_year, @max_year)")
    # emi_df.to_csv(output_path, index=False)
    return emi_df


# %% STEP 2. Set Input/Output Paths

# INPUT PATHS
inventory_workbook_path = ghgi_data_dir_path / "WW_State-level Estimates_90-22_27June2024.xlsx"

# OUTPUT PATHS
output_path_ww_ad_emi = emi_data_dir_path / "ww_ad_emi.csv"
output_path_ww_centralaerobic_emi = emi_data_dir_path / "ww_centralaerobic_emi.csv"
output_path_ww_centralanaerobic_emi = emi_data_dir_path / "ww_centralanaerobic_emi.csv"
output_path_ww_septic_emi = emi_data_dir_path / "ww_septic_emi.csv"
output_path_ww_brew_emi = emi_data_dir_path / "ww_brew_emi.csv"
output_path_ww_ethanol_emi = emi_data_dir_path / "ww_ethanol_emi.csv"
output_path_ww_fv_emi = emi_data_dir_path / "ww_fv_emi.csv"
output_path_ww_mp_emi = emi_data_dir_path / "ww_mp_emi.csv"
output_path_ww_petrref_emi = emi_data_dir_path / "ww_petrref_emi.csv"
output_path_ww_pp_emi = emi_data_dir_path / "ww_pp_emi.csv"

# %% STEP 3. Function Calls

# get_ww_inv_data(
#     inventory_workbook_path,
#     output_path_ww_ad_emi,
#     "ww_ad_emi"
# )
# get_ww_inv_data(
#     inventory_workbook_path,
#     output_path_ww_centralaerobic_emi,
#     "ww_centralaerobic_emi"
# )
# get_ww_inv_data(
#     inventory_workbook_path,
#     output_path_ww_centralanaerobic_emi,
#     "ww_centralanaerobic_emi"
# )
# get_ww_inv_data(
#     inventory_workbook_path,
#     output_path_ww_septic_emi,
#     "ww_septic_emi"
# )
get_ww_inv_data(
    inventory_workbook_path,
    output_path_ww_brew_emi,
    "ww_brew_emi"
)
get_ww_inv_data(
    inventory_workbook_path,
    output_path_ww_ethanol_emi,
    "ww_ethanol_emi"
)
get_ww_inv_data(
    inventory_workbook_path,
    output_path_ww_fv_emi,
    "ww_fv_emi"
)
testing = get_ww_inv_data(
    inventory_workbook_path,
    output_path_ww_mp_emi,
    "ww_mp_emi"
)
get_ww_inv_data(
    inventory_workbook_path,
    output_path_ww_petrref_emi,
    "ww_petrref_emi"
)
get_ww_inv_data(
    inventory_workbook_path,
    output_path_ww_pp_emi,
    "ww_pp_emi"
)

# %% YEAR TESTING

inventory_workbook_path = ghgi_data_dir_path / "WW_State-level Estimates_90-22_27June2024.xlsx"


emi_df = (
    # read in the data
    pd.read_excel(
        inventory_workbook_path,
        sheet_name="M_P Emissions",
        skiprows=82,
        nrows=56,
        # usecols="D:AL",
        index_col=None
    )
    )

emi_df = emi_df.dropna(how="all", axis=1)

emi_df.columns.values[0] = "State"
emi_df.columns.values[1] = "Metric"
emi_df.columns.values[2:] = list(range(1990, 2023))
emi_df = emi_df.drop(columns="Metric")


columns_to_keep = [col for col in emi_df.columns if not str(col).isdigit() or int(col) <= max_year]
emi_df = emi_df[columns_to_keep]

# An alternative option may be to NOT use "usecols" arg, dropna(how="all", axis=1), and then filter for the columns you need up front (after lowercasing column names)



emi_df = emi_df.rename(columns=lambda x: str(x).lower()) \
    .set_index("state") \
    .apply(pd.to_numeric, errors="coerce") \
    .dropna(how="all") \
    .replace(0, pd.NA) \
    .dropna(how="all")

