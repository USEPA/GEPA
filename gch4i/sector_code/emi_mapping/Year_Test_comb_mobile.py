
"""
Name:                   comb_mobile_emis.py
Date Last Modified:     2024-07-10
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of combustion - mobile emissions to State, Year,
                        emissions format
Input Files:            - SIT Mobile Dataframe 5.24.2023.xlsx
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
from gch4i.utils import tg_to_kt  # Determine if this conversion rate is correct


# %% STEP 1. Create Emi Mapping Functions
def get_comb_mobile_inv_data(input_path, output_path, subcategory):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - comb_mob_aircraft_emi
    - comb_mob_alt_fuel_emi
    - comb_mob_buses_emi
    - comb_mob_con_equip_emi
    - comb_mob_diesel_emi"
    - comb_mob_farm_equip_emi
    - comb_mob_ldt_emi
    - comb_mob_loco_emi
    - comb_mob_mcycle_emi
    - comb_mob_other_emi
    - comb_mob_pass_emi
    - comb_mob_vehicles_emi
    - comb_mob_waterways_emi
    - comb_mob_non_hwy_emi
    - comb_mob_gas_hwy_emi
    """
    subcategory_strings = {
        "comb_mob_aircraft_emi": "Aircraft",
        "comb_mob_alt_fuel_emi": "Alternative Fuel Vehicles$",
        "comb_mob_buses_emi": "Buses",
        "comb_mob_con_equip_emi": "Construction Equipment",
        "comb_mob_diesel_emi": "Diesel Highway$",
        "comb_mob_farm_equip_emi": "Farm Equipment",
        "comb_mob_ldt_emi": "Light-Duty Trucks",
        "comb_mob_loco_emi": "Locomotives",
        "comb_mob_mcycle_emi": "Motorcycles",
        "comb_mob_other_emi": "Other",
        "comb_mob_pass_emi": "Passenger Cars",
        "comb_mob_vehicles_emi": "Vehicles",
        "comb_mob_waterways_emi": "Boats",
        "comb_mob_non_hwy_emi": "Non-Highway$",
        "comb_mob_gas_hwy_emi": "Gasoline Highway$"
        }
    subcategory_string = subcategory_strings.get(
        subcategory
    )
    if subcategory_string is None:
        raise ValueError("""Invalid arg. Please use one of the following arguments:
                comb_mob_aircraft_emi, comb_mob_alt_fuel_emi, comb_mob_con_equip_emi,
                comb_mob_diesel_emi, comb_mob_farm_equip_emi, comb_mob_ldt_emi.,
                comb_mob_loco_emi, comb_mob_mcycle_emi, comb_mob_other_emi,
                comb_mob_pass_emi, comb_mob_waterways_emi, comb_mob_non_hwy_emi,
                comb_mob_gas_hwy_emi""")
    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="Table",
            nrows=3112,
            usecols="A:AI",
            index_col=None
        )
    )
    emi_df2 = (emi_df.rename(columns=lambda x: str(x).lower())
        .drop(columns=["state"])
        .rename(columns={'unnamed: 0': 'state_code'})
        # Remove CO2 from sector and get emissions for specific subcategory
        .query('sector.str.contains("CH4") == True' and f'sector.str.contains("{subcategory_string}", regex={"False" if subcategory_string in ["comb_mob_alt_fuel_emi", "comb_mob_diesel_emi", "comb_mob_non_hwy_emi", "comb_mob_gas_hwy_emi"] else "True"})', engine='python')
        # change sector inputs to unified subcategory
        .drop(columns=["sector"])
        # Drop rows with 0 across all years
        .replace(0, pd.NA)
        .dropna(subset=emi_df.columns[emi_df.columns.get_loc('1990'):], how='all')
        .fillna(0)
        # reset the index state back to a column
        .reset_index(drop=True)
        # make table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # correct column types
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # group by state and year and sum the ch4_tg column
        .groupby(["state_code", "year"]).sum().reset_index()
        # filter to required years
        .query("year.between(@min_year, @max_year)")
        .sort_values("year")
    )
    emi_df2.to_csv(output_path, index=False)
    # return emi_df2

# %% STEP 2. Set Input/Output Paths
# INPUT PATHS
inventory_workbook_path = ghgi_data_dir_path / "SIT Mobile Dataframe 5.24.2023.xlsx"
# outpath = tmp_data_dir_path / "comb_mobile_emis_test.csv"

# OUTPUT PATHS
output_path_comb_mob_aircraft_emi = V3_DATA_PATH / "emis/comb_mob_aircraft_emi.csv"
output_path_comb_mob_alt_fuel_emi = V3_DATA_PATH / "emis/comb_mob_alt_fuel_emi.csv"
output_path_comb_mob_buses_emi = V3_DATA_PATH / "emis/comb_mob_buses_emi.csv"
output_path_comb_mob_con_equip_emi = V3_DATA_PATH / "emis/comb_mob_con_equip_emi.csv"
output_path_comb_mob_diesel_emi = V3_DATA_PATH / "emis/comb_mob_diesel_emi.csv"
output_path_comb_mob_farm_equip_emi = V3_DATA_PATH / "emis/comb_mob_farm_equip_emi.csv"
output_path_comb_mob_ldt_emi = V3_DATA_PATH / "emis/comb_mob_ldt_emi.csv"
output_path_comb_mob_loco_emi = V3_DATA_PATH / "emis/comb_mob_loco_emi.csv"
output_path_comb_mob_mcycle_emi = V3_DATA_PATH / "emis/comb_mob_mcycle_emi.csv"
output_path_comb_mob_other_emi = V3_DATA_PATH / "emis/comb_mob_other_emi.csv"
output_path_comb_mob_pass_emi = V3_DATA_PATH / "emis/comb_mob_pass_emi.csv"
output_path_comb_mob_vehicles_emi = V3_DATA_PATH / "emis/comb_mob_vehicles_emi.csv"
output_path_comb_mob_waterways_emi = V3_DATA_PATH / "emis/comb_mob_waterways_emi.csv"
output_path_comb_mob_non_hwy_emi = V3_DATA_PATH / "emis/comb_mob_non_hwy_emi.csv"
output_path_comb_mob_gas_hwy_emi = V3_DATA_PATH / "emis/comb_mob_gas_hwy_emi.csv"

# %% STEP 3. Function Calls
# Aircraft
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_aircraft_emi,
    "comb_mob_aircraft_emi"
)
# Alternative Fuel Vehicles
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_alt_fuel_emi,
    "comb_mob_alt_fuel_emi"
    )
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_buses_emi,
    "comb_mob_buses_emi"
    )
# Construction Equipment
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_con_equip_emi,
    "comb_mob_con_equip_emi"
    )
# Diesel Highway
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_diesel_emi,
    "comb_mob_diesel_emi"
    )
# Farm Equipment
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_farm_equip_emi,
    "comb_mob_farm_equip_emi"
    )
# Light-Duty Trucks
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_ldt_emi,
    "comb_mob_ldt_emi"
    )
# Locomotives
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_loco_emi,
    "comb_mob_loco_emi"
    )
# Motorcycles
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_mcycle_emi,
    "comb_mob_mcycle_emi"
    )
# Other
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_other_emi,
    "comb_mob_other_emi"
    )
# Passenger Cars
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_pass_emi,
    "comb_mob_pass_emi"
    )
# Vehicles
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_vehicles_emi,
    "comb_mob_vehicles_emi"
    )
# Waterways
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_waterways_emi,
    "comb_mob_waterways_emi"
    )
# Non-Highway
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_non_hwy_emi,
    "comb_mob_non_hwy_emi"
    )
# Gasoline Highway
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_gas_hwy_emi,
    "comb_mob_gas_hwy_emi"
    )

# %% STEP TEST. Test Function Calls
testing = get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_vehicles_emi,
    "comb_mob_vehicles_emi"
    )

# %% TEST FOR YEAR PULL

inventory_workbook_path = ghgi_data_dir_path / "SIT Mobile Dataframe 5.24.2023.xlsx"


emi_df = (
    # read in the data
    pd.read_excel(
        inventory_workbook_path,
        sheet_name="Table",
        nrows=3112,
        # usecols="A:AI",
        index_col=None
    )
    )

first_row = emi_df.columns
year_columns = [col for col in first_row if str(col).isdigit()]
years = [int(col) for col in year_columns]

df_filtered = emi_df.loc[:, emi_df.columns.astype(str) <= str(max_year)]

columns_to_keep = [col for col in emi_df.columns if not str(col).isdigit() or int(col) <= max_year]
emi_df = emi_df[columns_to_keep]


# %%
