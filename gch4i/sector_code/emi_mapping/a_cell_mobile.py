"""
Name:                   a_excel_dict_mobile.py
Date Last Modified:     2024-08-12
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of combustion - mobile emissions to State, Year,
                        emissions format
Input Files:            - Mobile non-CO2 InvDB State Breakout_2022.xlsx
                        - SIT Mobile Dataframe 5.24.2023.xlsx
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  - This version of emi mapping is draft for mapping .py files
                        - Relative proportions from "SIT Mobile",
                        Emissions numbers from "Mobile non-CO2 InvDB".
"""

# Change return emi_df3 in all 3 spaces, not just at the end

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
import pandas as pd
import ast
from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    emi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import tg_to_kt
from gch4i.sector_code.emi_mapping.a_excel_dict import (read_excel_dict_cell, file_path)


# %% STEP 1. Create Emi Mapping Functions
def get_comb_mobile_inv_data(input_path, output_path, subcategory):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - Emi_Passenger
    - Emi_Light
    - Emi_Heavy
    - Emi_AllRoads
    - Emi_Waterways
    - Emi_Railroads
    - Emi_Aircraft
    - Emi_Farm
    - Emi_Equip
    - Emi_Other
    """

    subcategory_strings = read_excel_dict_cell(file_path, emission="Mobile_Comb", sheet="Single_Cell")

    # subcategory_strings = {
    #    "Emi_Passenger": ["Gasoline", "Gasoline", "Cars|Motorcycles"],
    #    "Emi_Light": ["Gasoline Highway", "Gasoline Highway", "Light-Duty"],
    #    "Emi_Heavy": ["Gasoline|Diesel", "Gasoline Highway", "Heavy-Duty"],  # ifelse
    #    "Emi_AllRoads": ["Alternative"],  # Stop at emi_df
    #    "Emi_Waterways": ["Non-Highway$", "Non-Highway", "Boats"],
    #    "Emi_Railroads": ["Non-Highway$", "Non-Highway", "Locomotives"],
    #    "Emi_Aircraft": ["Non-Highway$", "Non-Highway", "Aircraft"],
    #    "Emi_Farm": ["Farm Equipment"],  # Stop at emi_df
    #    "Emi_Equip": ["Construction"],  # Stop at emi_df
    #    "Emi_Other": ["Mobile Non-Highway Other"]  # Stop at emi_df
    #    }

    subcategory_string = subcategory_strings.get(
        subcategory
    )
    if subcategory_string is None:
        raise ValueError("""Invalid arg. Please use one of the following arguments:
                Emi_Passenger, Emi_Light, Emi_Heavy, Emi_AllRoads, Emi_Waterways,
                Emi_Railroads, Emi_Aircraft, Emi_Farm, Emi_Equip, Emi_Other""")

    # Follow this path if subcategory is Emi_Heavy
    if subcategory == "Emi_Heavy":
        emi_df = (
        # read in the data
        pd.read_excel(
            inventory_workbook_path,
            sheet_name="InvDB",
            skiprows=15,
            index_col=None
            )
        )
        # Remove unnecessary columns
        columns_to_keep = [col for col in emi_df.columns if not str(col).isdigit() or int(col) <= max_year]
        emi_df = emi_df[columns_to_keep]

        emi_df = (
            emi_df.rename(columns=lambda x: str(x).lower())
            .filter(regex="subcategory1|georef|ghg|19|20")
            .rename(columns={"georef": "state"})
            .query('ghg == "CH4"')
            .query(f'subcategory1.str.contains("{subcategory_string[0]}", regex=True)', engine='python')  # dict[0]
            .drop(columns=["ghg"])
            .set_index(["state", "subcategory1"])
            .replace(0, pd.NA)
            .apply(pd.to_numeric, errors="coerce")
            .dropna(how="all")
            .fillna(0)
            .reset_index()
            .melt(id_vars=["state", "subcategory1"], var_name="year", value_name="ch4_tg")
            .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
            .drop(columns=["ch4_tg"])
            .astype({"year": int, "ch4_kt": float})
            .fillna({"ch4_kt": 0})
            .groupby(["state", "subcategory1", "year"]).sum().reset_index()
            .query("year.between(@min_year, @max_year-1)")  # usually max_year, missing 2022
            .pivot(index=["state", "year"], columns="subcategory1", values="ch4_kt").reset_index()
            .sort_values(["state", "year"])
        )

        ################################################################################

        emi_df2 = (
            # read in the data
            pd.read_excel(
                inventory_workbook_path2,
                sheet_name="Table",
                index_col=None
            )
        )

        emi_df2 = (
            emi_df2.rename(columns=lambda x: str(x).lower())
            .drop(columns=["state"])
            .rename(columns={'unnamed: 0': 'state'})
            # Remove CO2 from sector and get emissions for specific subcategory
            .query(f'sector.str.contains("CH4") and sector.str.contains("{subcategory_string[1]}")', engine='python')  # dict[1]
            .query(f'sector.str.contains("{subcategory_string[2]}", regex=True) or sector.str.endswith("{subcategory_string[1]}")', engine='python')   # dict[2], [1]
            .melt(id_vars=["state", "sector"], var_name="year", value_name="ch4_kt")
            .astype({"year": int})
            .query("year.between(@min_year, @max_year)")
            .pivot_table(index=["state", "year"], columns="sector", values="ch4_kt")
        )

        emi_df2 = (
            emi_df2.div(emi_df2.iloc[:, 0], axis=0)
            .drop(columns=emi_df2.columns[0])
            .rename(columns={"Energy - Mobile Combustion - CH4 - Gasoline Highway - Heavy-Duty Vehicles": "Proportion"})
            .reset_index()
        )

        ################################################################################

        emi_df3 = (
            pd.merge(emi_df, emi_df2, on=["state", "year"], how="left")
            .assign(new_gas=lambda df: df["Gasoline Highway"] * df["Proportion"])
            .assign(ch4_kt=lambda df: df["Diesel Highway"] + df["new_gas"])
            .drop(columns=["Gasoline Highway", "Proportion", "Diesel Highway", "new_gas"])
        )
        # emi_df3.to_csv(output_path, index=False)
        return emi_df3

    # Follow this path if subcategory is not Emi_Heavy
    else:
        emi_df = (
            # read in the data
            pd.read_excel(
                inventory_workbook_path,
                sheet_name="InvDB",
                skiprows=15,
                index_col=None
            )
        )

        # Remove unnecessary columns
        columns_to_keep = [col for col in emi_df.columns if not str(col).isdigit() or int(col) <= max_year]
        emi_df = emi_df[columns_to_keep]

        emi_df = (
            emi_df.rename(columns=lambda x: str(x).lower())
            .filter(regex="subcategory1|georef|ghg|19|20")
            .rename(columns={"georef": "state"})
            .query('ghg == "CH4"')
            .query(f'subcategory1.str.contains("{subcategory_string[0]}", regex=True)', engine='python')  # dict[0]
            .drop(columns=["ghg", "subcategory1"])
            .set_index("state")
            .replace(0, pd.NA)
            .apply(pd.to_numeric, errors="coerce")
            .dropna(how="all")
            .fillna(0)
            .reset_index()
            .melt(id_vars="state", var_name="year", value_name="ch4_tg")
            .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
            .drop(columns=["ch4_tg"])
            .astype({"year": int, "ch4_kt": float})
            .fillna({"ch4_kt": 0})
            .groupby(["state", "year"]).sum().reset_index()
            .query("year.between(@min_year, @max_year-1)")  # usually max_year, missing 2022
            .sort_values("year")
        )
        ################################################################################
        # End Here if subcategory is in list
        if subcategory in (["Emi_AllRoads", "Emi_Farm", "Emi_Equip", "Emi_Other"]):
            return emi_df
        ################################################################################

        emi_df2 = (
            # read in the data
            pd.read_excel(
                inventory_workbook_path2,
                sheet_name="Table",
                index_col=None
            )
        )

        emi_df2 = (
            emi_df2.rename(columns=lambda x: str(x).lower())
            .drop(columns=["state"])
            .rename(columns={'unnamed: 0': 'state'})
            # Remove CO2 from sector and get emissions for specific subcategory
            .query(f'sector.str.contains("CH4") and sector.str.contains("{subcategory_string[1]}", regex=True)', engine='python')  # dict[0]
            .query(f'sector.str.contains("{subcategory_string[2]}", regex=True) or sector.str.endswith("{subcategory_string[1]}")')   # dict[1], [0]
            .melt(id_vars=["state", "sector"], var_name="year", value_name="ch4_kt")
            .astype({"year": int})
            .query("year.between(@min_year, @max_year)")
            .pivot_table(index=["state", "year"], columns="sector", values="ch4_kt")
        )

        emi_df2 = (
            emi_df2.div(emi_df2.iloc[:, 0], axis=0)
            .drop(columns=emi_df2.columns[0])
            .assign(proportion=lambda df: df.sum(axis=1))
            .iloc[:, -1]
            .reset_index()
        )

########################################################################################

        emi_df3 = (
            pd.merge(emi_df, emi_df2, on=["state", "year"], how="left")
            .assign(ch4_kt=lambda df: df["ch4_kt"] * df["proportion"])
            .drop(columns=["proportion"])
        )
        # emi_df3.to_csv(output_path, index=False)
        return emi_df3


# %% STEP 2. Set Input/Output Paths
# INPUT PATHS
inventory_workbook_path = ghgi_data_dir_path / "combustion_mobile/Mobile non-CO2 InvDB State Breakout_2022.xlsx"
inventory_workbook_path2 = ghgi_data_dir_path / "combustion_mobile/SIT Mobile Dataframe 5.24.2023.xlsx"



# OUTPUT PATHS
output_path_emi_passenger = emi_data_dir_path / "emi_passenger.csv"
output_path_emi_light = emi_data_dir_path / "emi_light.csv"
output_path_emi_heavy = emi_data_dir_path / "emi_heavy.csv"
output_path_emi_allroads = emi_data_dir_path / "emi_allroads.csv"
output_path_emi_waterways = emi_data_dir_path / "emi_waterways.csv"
output_path_emi_railroads = emi_data_dir_path / "emi_railroads.csv"
output_path_emi_aircraft = emi_data_dir_path / "emi_aircraft.csv"
output_path_emi_farm = emi_data_dir_path / "emi_farm.csv"
output_path_emi_equip = emi_data_dir_path / "emi_equip.csv"
output_path_emi_other = emi_data_dir_path / "emi_other.csv"


# %% STEP 3. Function Calls

get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_passenger,
    "Emi_Passenger"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_light,
    "Emi_Light"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_heavy,
    "Emi_Heavy"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_allroads,
    "Emi_AllRoads"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_waterways,
    "Emi_Waterways"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_railroads,
    "Emi_Railroads"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_aircraft,
    "Emi_Aircraft"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_farm,
    "Emi_Farm"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_equip,
    "Emi_Equip"
)
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_other,
    "Emi_Other"
)

# %% TESTING AGAIN

testing = get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_emi_other,
    "Emi_Light"
)

# %% AGAIN
testing = read_excel_dict_cell(file_path, sheet="Single_Cell", emission="Wastewater")


df = pd.read_excel(file_path, sheet_name="Single_Cell")

# Filter for Emissions Dictionary
df = df.loc[df['Emission'] == "Mobile_Comb"].drop(columns=['Emission'])

# Assign to object

new_dict = ast.literal_eval(df.iloc[0, 0])
