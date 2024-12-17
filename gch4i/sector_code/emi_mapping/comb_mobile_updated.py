"""
Name:                   comb_mobile_updated.py
Date Last Modified:     2024-08-07
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of combustion - mobile emissions to State, Year,
                        emissions format
Input Files:            - Mobile non-CO2 InvDB State Breakout_2022.xlsx
                        - SIT Mobile Dataframe 5.24.2023.xlsx
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  - This version of emi mapping is draft for mapping .py files
                        - Missing year 2022 from SIT Mobile Dataframe 5.24.2023.xlsx
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
    subcategory_strings = {
        "Emi_Passenger": ["Gasoline Highway", "Cars|Motorcycles"],
        "Emi_Light": ["Gasoline", "Light-Duty"],
        "Emi_Heavy": ["Gasoline|Diesel", "Gasoline Highway", "Heavy-Duty"],  # ifelse
        "Emi_AllRoads": ["Alternative"],  # Stop at emi_df
        "Emi_Waterways": ["Non-Highway$", "Boats"],
        "Emi_Railroads": ["Non-Highway$", "Locomotives"],
        "Emi_Aircraft": ["Non-Highway\$", "Aircraft"],
        "Emi_Farm": ["Mobile Non-Highway Farm Equipment"],  # Stop at emi_df
        "Emi_Equip": ["Mobile Non-Highway Construction"],  # Stop at emi_df
        "Emi_Other": ["Mobile Non-Highway Other"]  # Stop at emi_df
        }
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
            .query(f'subcategory1.str.contains({subcategory_string[0]}, regex=True) == True', engine='python')  # dict[0]
            .drop(columns=["ghg"])
            .set_index(["state", "subcategory1"])
            .replace(0, pd.NA)
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
            .query(f'sector.str.contains("CH4") == True and sector.str.contains({subcategory_string[1]})', engine='python')  # dict[1]
            .query(f'sector.str.contains({subcategory_string[2]}, regex=True) or sector.str.endswith({subcategory_string[1]})', engine='python')   # dict[2], [1]
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
            .query(f'subcategory1.str.contains({subcategory_string[0]}, regex=True) == True', engine='python')  # dict[0]
            .drop(columns=["ghg", "subcategory1"])
            .set_index("state")
            .replace(0, pd.NA)
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
        if subcategory.isin(["Emi_AllRoads", "Emi_Farm", "Emi_Equip", "Emi_Other"]):
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
            .query(f'sector.str.contains("CH4") == True and sector.str.contains({subcategory_string[0]})', engine='python')  # dict[0]
            .query(f'sector.str.contains({subcategory_string[1]}, regex=True) or sector.str.endswith({subcategory_string[0]})')   # dict[1], [0]
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
inventory_workbook_path2 = ghgi_data_dir_path / "SIT Mobile Dataframe 5.24.2023.xlsx"
# outpath = tmp_data_dir_path / "comb_mobile_emis_test.csv"


# OUTPUT PATHS
output_path_comb_mob_aircraft_emi = V3_DATA_PATH / "emis/comb_mob_aircraft_emi.csv"


# %% STEP 3. Function Calls
# Aircraft
get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_aircraft_emi,
    "Emi_Passenger"
)










    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="Table",
            nrows=3112,
            index_col=None
        )
    )
    emi_df2 = (emi_df.rename(columns=lambda x: str(x).lower())
        .drop(columns=["state"])
        .rename(columns={'unnamed: 0': 'state_code'})
        # Remove CO2 from sector and get emissions for specific subcategory
        #.query('sector.str.contains("CH4") == True and sector.str.contains("|".join(subcategory_string), regex=True)', engine='python')
        .query('sector.str.contains("CH4") == True and sector.str.contains("{0}") and sector.str.contains("{1}"), regex=True)'.format(subcategory_string[0], subcategory_string[1]))
        # change sector inputs to unified subcategory
        .drop(columns=["sector"])
        # Drop rows with 0 across all years
        .replace(0, pd.NA)
        .set_index("state_code")
        .dropna(how="all")
        .fillna(0)
        # reset the index state back to a column
        .reset_index()
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
    return emi_df2
    # emi_df2.to_csv(output_path, index=False)




# %% TESTING
########################################################################################

"""
Subcategory1 filter name
list of index loacations for which fuel_types to keep

if Emi_AllRoads, Emi_Farm, Emi_Equip, Emi_Other, return emi_df (don't need emi_df2 or emi_df3)
if Emi_Heavy, grab Diesel and Gasoline emissions, calculate proportion of vehicle/gasoline, multiply by Gas, add together
"""

subcategory_strings = {
        "Emi_Passenger": ["Gasoline Highway", "Cars|Motorcycles"],
        "Emi_Light": ["Gasoline", "Light-Duty"],
        "Emi_Heavy": ["Gasoline|Diesel", "Gasoline Highway", "Heavy-Duty"],  # ifelse
        "Emi_AllRoads": ["Alternative"],  # Stop at emi_df
        "Emi_Waterways": ["Non-Highway$", "Boats"],
        "Emi_Railroads": ["Non-Highway$", "Locomotives"],
        "Emi_Aircraft": ["Non-Highway$", "Aircraft"],
        "Emi_Farm": ["Mobile Non-Highway Farm Equipment"],  # Stop at emi_df
        "Emi_Equip": ["Mobile Non-Highway Construction"],  # Stop at emi_df
        "Emi_Other": ["Mobile Non-Highway Other"]  # Stop at emi_df
        }

########################################################################################

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
    .query('subcategory1.str.contains("Gasoline") == True', engine='python')  # dict[0]
    .drop(columns=["ghg", "subcategory1"])
    .set_index("state")
    .replace(0, pd.NA)
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

########################################################################################
if subcategory.isin(["Emi_AllRoads", "Emi_Farm", "Emi_Equip", "Emi_Other"]):
    return emi_df
########################################################################################

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
    .query('sector.str.contains("CH4") == True and sector.str.contains("Gasoline Highway")', engine='python')  # dict[0]
    .query('sector.str.contains("Cars|Motorcycles", regex=True) or sector.str.endswith("Gasoline Highway")')   # dict[1], [0]
    .melt(id_vars=["state","sector"], var_name="year", value_name="ch4_kt")
    .astype({"year": int})
    .query("year.between(@min_year, @max_year)")
    .pivot_table(index=["state","year"], columns="sector", values="ch4_kt")
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

########################################################################################
########################################################################################
# Emi_Heavy Path
########################################################################################

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
    .query('subcategory1.str.contains("Gasoline|Diesel") == True', engine='python')  # dict[0]
    .drop(columns=["ghg"])
    .set_index(["state", "subcategory1"])
    .replace(0, pd.NA)
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

########################################################################################

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
    .query('sector.str.contains("CH4") == True and sector.str.contains("Gasoline Highway")', engine='python')  # dict[1]
    .query('sector.str.contains("Heavy-Duty", regex=True) or sector.str.endswith("Gasoline Highway")')   # dict[2], [1]
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

########################################################################################

emi_df3 = (
    pd.merge(emi_df, emi_df2, on=["state", "year"], how="left")
    .assign(new_gas=lambda df: df["Gasoline Highway"] * df["Proportion"])
    .assign(ch4_kt=lambda df: df["Diesel Highway"] + df["new_gas"])
    .drop(columns=["Gasoline Highway", "Proportion", "Diesel Highway", "new_gas"])
)

########################################################################################
















get_comb_mobile_inv_data(
    inventory_workbook_path,
    output_path_comb_mob_aircraft_emi,
    "Emi_Passenger"
)
