
"""
Name:                   comb_stationary_emis.py
Date Last Modified:     2024-07-10
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of combustion - stationary emissions to State, Year,
                        emissions format
Input Files:            - Stationary Calcs 90-22_3_12_24_FR.xlsx
Output Files:           - TBD
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
from gch4i.utils import tg_to_kt  # Determine if this conversion rate is correct


# %% STEP 1. Create Emi Mapping Functions
def get_comb_stat_inv_data(input_path, output_path, subcategory):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - comb_stat_coal_elec_emi
    - comb_stat_coal_indu_emi
    - comb_stat_coal_comm_emi
    - comb_stat_coal_resi_emi
    - comb_stat_coal_nomap_emi
    - comb_stat_oil_elec_emi
    - comb_stat_oil_indu_emi
    - comb_stat_oil_comm_emi
    - comb_stat_oil_resi_emi
    - comb_stat_oil_nomap_emi
    - comb_stat_gas_elec_emi
    - comb_stat_gas_indu_emi
    - comb_stat_gas_comm_emi
    - comb_stat_gas_resi_emi
    - comb_stat_gas_nomap_emi
    - comb_stat_wood_elec_emi
    - comb_stat_wood_indu_emi
    - comb_stat_wood_comm_emi
    - comb_stat_wood_resi_emi
    - comb_stat_wood_nomap_emi
    """
    subcategory_strings = {
        "comb_stat_coal_elec_emi": ["Electricity Generation", "Coal"],
        "comb_stat_coal_indu_emi": ["Industrial", "Coal"],
        "comb_stat_coal_comm_emi": ["Commercial", "Coal"],
        "comb_stat_coal_resi_emi": ["Residential", "Coal"],
        "comb_stat_coal_nomap_emi": ["US Territories", "Coal"],
        "comb_stat_oil_elec_emi": ["Electricity Generation", "Petroleum"],
        "comb_stat_oil_indu_emi": ["Industrial", "Petroleum"],
        "comb_stat_oil_comm_emi": ["Commercial", "Petroleum"],
        "comb_stat_oil_resi_emi": ["Residential", "Petroleum"],
        "comb_stat_oil_nomap_emi": ["US Territories", "Petroleum"],
        "comb_stat_gas_elec_emi": ["Electricity Generation", "Natural Gas"],
        "comb_stat_gas_indu_emi": ["Industrial", "Natural Gas"],
        "comb_stat_gas_comm_emi": ["Commercial", "Natural Gas"],
        "comb_stat_gas_resi_emi": ["Residential", "Natural Gas"],
        "comb_stat_gas_nomap_emi": ["US Territories", "Natural Gas"],
        "comb_stat_wood_elec_emi": ["Electricity Generation", "Biomass"],
        "comb_stat_wood_indu_emi": ["Industrial", "Biomass"],
        "comb_stat_wood_comm_emi": ["Commercial", "Biomass"],
        "comb_stat_wood_resi_emi": ["Residential", "Biomass"],
        "comb_stat_wood_nomap_emi": ["US Territories", "Biomass"]
        }
    subcategory_string = subcategory_strings.get(subcategory)
    if subcategory_string is None:
        raise ValueError("""Invalid arg. Please choose from the following arguments:
            comb_stat_coal_elec_emi, comb_stat_coal_indu_emi, comb_stat_coal_comm_emi,
            comb_stat_coal_resi_emi, comb_stat_coal_nomap_emi, comb_stat_oil_elec_emi,
            comb_stat_oil_indu_emi, comb_stat_oil_comm_emi, comb_stat_oil_resi_emi,
            comb_stat_oil_nomap_emi, comb_stat_gas_elec_emi, comb_stat_gas_indu_emi,
            comb_stat_gas_comm_emi, comb_stat_gas_resi_emi, comb_stat_gas_nomap_emi,
            comb_stat_wood_elec_emi, comb_stat_wood_indu_emi, comb_stat_wood_comm_emi,
            comb_stat_wood_resi_emi, comb_stat_wood_nomap_emi""")
    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=41,
            usecols="A:BA",
            index_col=None
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # create state_code column
        .rename(columns={'georef': 'state_code'})
        # set state code as index
        .set_index("state_code")
        # query to filter ghg == CH4 & category & fuel1
        .query('(ghg == "CH4") & (category == @subcategory_string[0]) & (fuel1 == @subcategory_string[1])')
        # filter columns
        .filter(regex='state_code|19|20')
        # convert "NO" and "NE" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
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
    # emi_df.to_csv(output_path, index=False)
    return emi_df

# %% STEP 2. Set Input/Output Paths


# INPUT PATHS
inventory_workbook_path = ghgi_data_dir_path / "Stationary Calcs 90-22_3_12_24_FR.xlsx"

# OUTPUT PATHS
output_path_comb_stat_coal_elec_emi = V3_DATA_PATH / "emis/comb_stat_coal_elec_emi.csv"
output_path_comb_stat_coal_indu_emi = V3_DATA_PATH / "emis/comb_stat_coal_indu_emi.csv"
output_path_comb_stat_coal_comm_emi = V3_DATA_PATH / "emis/comb_stat_coal_comm_emi.csv"
output_path_comb_stat_coal_resi_emi = V3_DATA_PATH / "emis/comb_stat_coal_resi_emi.csv"
output_path_comb_stat_coal_nomap_emi = V3_DATA_PATH / "emis/comb_stat_coal_nomap_emi.csv"
output_path_comb_stat_oil_elec_emi = V3_DATA_PATH / "emis/comb_stat_oil_elec_emi.csv"
output_path_comb_stat_oil_indu_emi = V3_DATA_PATH / "emis/comb_stat_oil_indu_emi.csv"
output_path_comb_stat_oil_comm_emi = V3_DATA_PATH / "emis/comb_stat_oil_comm_emi.csv"
output_path_comb_stat_oil_resi_emi = V3_DATA_PATH / "emis/comb_stat_oil_resi_emi.csv"
output_path_comb_stat_oil_nomap_emi = V3_DATA_PATH / "emis/comb_stat_oil_nomap_emi.csv"
output_path_comb_stat_gas_elec_emi = V3_DATA_PATH / "emis/comb_stat_gas_elec_emi.csv"
output_path_comb_stat_gas_indu_emi = V3_DATA_PATH / "emis/comb_stat_gas_indu_emi.csv"
output_path_comb_stat_gas_comm_emi = V3_DATA_PATH / "emis/comb_stat_gas_comm_emi.csv"
output_path_comb_stat_gas_resi_emi = V3_DATA_PATH / "emis/comb_stat_gas_resi_emi.csv"
output_path_comb_stat_gas_nomap_emi = V3_DATA_PATH / "emis/comb_stat_gas_nomap_emi.csv"
output_path_comb_stat_wood_elec_emi = V3_DATA_PATH / "emis/comb_stat_wood_elec_emi.csv"
output_path_comb_stat_wood_indu_emi = V3_DATA_PATH / "emis/comb_stat_wood_indu_emi.csv"
output_path_comb_stat_wood_comm_emi = V3_DATA_PATH / "emis/comb_stat_wood_comm_emi.csv"
output_path_comb_stat_wood_resi_emi = V3_DATA_PATH / "emis/comb_stat_wood_resi_emi.csv"
output_path_comb_stat_wood_nomap_emi = V3_DATA_PATH / "emis/comb_stat_wood_nomap_emi.csv"

# %% STEP 3. Function Calls
# Coal
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_coal_elec_emi,
    "comb_stat_coal_elec_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_coal_indu_emi,
    "comb_stat_coal_indu_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_coal_comm_emi,
    "comb_stat_coal_comm_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_coal_resi_emi,
    "comb_stat_coal_resi_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_coal_nomap_emi,
    "comb_stat_coal_nomap_emi"
    )
# Oil
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_oil_elec_emi,
    "comb_stat_oil_elec_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_oil_indu_emi,
    "comb_stat_oil_indu_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_oil_comm_emi,
    "comb_stat_oil_comm_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_oil_resi_emi,
    "comb_stat_oil_resi_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_oil_nomap_emi,
    "comb_stat_oil_nomap_emi"
    )
# Gas
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_gas_elec_emi,
    "comb_stat_gas_elec_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_gas_indu_emi,
    "comb_stat_gas_indu_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_gas_comm_emi,
    "comb_stat_gas_comm_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_gas_resi_emi,
    "comb_stat_gas_resi_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_gas_nomap_emi,
    "comb_stat_gas_nomap_emi"
    )
# Wood
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_wood_elec_emi,
    "comb_stat_wood_elec_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_wood_indu_emi,
    "comb_stat_wood_indu_emi")
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_wood_comm_emi,
    "comb_stat_wood_comm_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_wood_resi_emi,
    "comb_stat_wood_resi_emi"
    )
get_comb_stat_inv_data(
    inventory_workbook_path,
    output_path_comb_stat_wood_nomap_emi,
    "comb_stat_wood_nomap_emi"
    )
