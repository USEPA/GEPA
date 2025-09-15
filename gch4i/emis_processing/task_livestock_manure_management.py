"""
Name:                   task_livestock_manure_management.py
Date Last Modified:     2025-09-15
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of Livestock Manure Management emissions to Year, Month,
                            State, County, emissions format
gch4i_name:             3B_manure_management
Input Files:            - {v3_ghgi_data_dir_path}/3B_manure_management/
                            Gridded Methane - Manure emissions by County_v1_17Sept2024
                            .xlsx
                        - {v4_ghgi_data_dir_path}/3B_manure_management/
                            Manure_CH4-N2O_Output_1990-2023_State.xlsx
Output Files:           - {emi_data_dir_path}/
                            manure_management_beef_emi.csv
                            manure_management_bison_emi.csv
                            manure_management_broilers_emi.csv
                            manure_management_chicken_emi.csv
                            manure_management_dairy_emi.csv
                            manure_management_goats_emi.csv
                            manure_management_horses_emi.csv
                            manure_management_layers_emi.csv
                            manure_management_mules_emi.csv
                            manure_management_pullets_emi.csv
                            manure_management_sheep_emi.csv
                            manure_management_swine_emi.csv
                            manure_management_turkeys_emi.csv
                            manure_management_onfeed_emi.csv
                            manure_management_cattle_emi.csv
Notes:                  - 2012-2022 emissions are calculated at the county level.
                        - 2023 county proportions are replicated from 2022 county
                            proportions.
                        - V4 Beef Cattle == V3 Beef + Cattle + OnFeed
                        - V4 Poultry == V3 Broilers + Chickens + Layers + Pullets +
                            Turkeys
                        - Other than V4 Beef Cattle and V4 Poultry, V3 and V4 align.
    - Emissions are calculated in 5 stages:
        1. Calculate 2012-2022 county-level emissions using V3 data
        2. Calculate 2022 county-level proportions using V3 data with V4 categories
        3. Read in 2023 state-level emissions using V4 data
        4. Apply 2022 county-level proportions to 2023 state-level emissions to get
            2023 county-level emissions
        5. Append 2012-2022 county-level emissions with 2023 county-level emissions
    - The 5 stage process uses county proportions from 2022 and disaggregates V4
        categories to V3 categories to construct consistent emissions across the
        2012-2023 period.
    - Testing code is provided at the end, if user would like to compare V3 emis,
        V4 inputs, and V4 emis.
"""
# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product, mark, task

import ast
import re

from gch4i.config import (
    V3_DATA_PATH,
    V4_DATA_PATH,
    v4_emi_data_dir_path,
    v3_ghgi_data_dir_path,
    v4_ghgi_data_dir_path,
    max_year,
    min_year,
)
from gch4i.utils import tg_to_kt

# %% Step 1. Create Function

# this is a refereence dictionary to map the GHGI subcategory names to the
# livestock categories used in gridded methane. These align with the queries written
# in the data guide.
# crosswalk_dict = {
#     "beef_NOF_bull": "Beef",
#     "beef_NOF_cow": "Beef",
#     "beef_NOF_heifers": "Cattle",
#     "beef_NOF_steer": "Cattle",
#     "beef_OF_heifers": "OnFeed",
#     "beef_OF_steer": "OnFeed",
#     "bison": "Bison",
#     "calf_NOF_beef": "Beef",
#     "calf_NOF_dairy": "Dairy",
#     "dairy_cow": "Dairy",
#     "dairy_heifers": "Dairy",
#     "goats": "Goats",
#     "horses": "Horses",
#     "mules": "Mules",
#     "poultry_broilers": "Broilers",
#     "poultry_chickens": "Chickens",
#     "poultry_layers": "Layers",
#     "poultry_pullets": "Pullets",
#     "poultry_turkeys": "Turkeys",
#     "sheep": "Sheep",
#     "swine_120_179": "Swine",
#     "swine_180": "Swine",
#     "swine_50": "Swine",
#     "swine_50_119": "Swine",
#     "swine_breeding": "Swine",
# }


def get_livestock_manure_management_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - Beef
        - beef_NOF_bull
        - beef_NOF_cow
        - calf_NOF_beef
    - Cattle
        - beef_NOF_steer
        - beef_NOF_heifers
    - Dairy
        - dairy_cow
        - dairy_heifers
        - calf_NOF_dairy
    - OnFeed
        - beef_OF_heifers
        - beef_OF_steer
    - Bison
    - Goats
    - Horses
    - Mules
    - Sheep
    - Swine
        - swine_50
        - swine_50_119
        - swine_120_179
        - swine_180
        - swine_breeding
    - Broilers
        - poultry_broilers
    - Layers
        - poultry_layers
    - Turkeys
        - poultry_turkeys
    - Chickens
        - poultry_chickens
    - Pullets
        - poultry_pullets

    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest
    params : dict
        additional parameters
    """

    # Read in data
    emi_df = pd.read_excel(
        in_path[0],
        sheet_name=params["arguments"][0],  # Sheet Name
        skiprows=params["arguments"][1],  # Skip Rows
    )

    # Copy emi_df as emi_df_base
    emi_df_base = emi_df.copy()

    # Establish query pattern
    substrings = params["substrings"]

    pattern = "|".join(re.escape(s) for s in substrings)

    # Fix year column names
    emi_df = emi_df.drop(emi_df.columns[0], axis=1)
    # Changed from Max year, since 2023 will be different
    emi_df.columns.values[5:] = list(range(min_year, 2022 + 1))

    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        # Remove AK and HI
        .query("state_code not in ['AK', 'HI']")
        # Filter for specific animal category
        .query(
            f'animal.str.contains(r"{pattern}", regex=True, na=False)',
            engine="python",
        )  # param
        .drop(columns=["animal"])
        .set_index(["state_code", "county", "fips", "month"])
        # Convert NA values to 0 & Drop states with no data
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state/county/fips/month
        .melt(
            id_vars=["state_code", "county", "fips", "month"],
            var_name="year",
            value_name="ch4_tg",
        )
        # Convert tg to kt
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # Ensure only years between min_year and max_year are included
        # Changed from max_year, since 2023 will be different
        .query("year.between(@min_year, 2022)")
        # Ensure state/county/fips/year/month grouping is unique
        # Fips kept in due to different counties having the same name
        .groupby(["state_code", "county", "fips", "year", "month"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
        .sort_values(by=["fips", "year", "month"])
        .reset_index()
    )

    ################################################################################
    # Calculate Proportions from 2022 data
    # Only Beef Cattle & Poultry will have a different V4 pattern than V3 pattern
    ################################################################################
    # Copy of emi_df_base
    emi_df2 = emi_df_base.copy()
    # Substrings for V4 pattern
    substrings2 = params["substrings2"]
    pattern2 = "|".join(re.escape(s) for s in substrings2)

    # Fix year column names
    emi_df2 = emi_df2.drop(emi_df2.columns[0], axis=1)
    # Changed from Max year, since 2023 will be different
    emi_df2.columns.values[5:] = list(range(min_year, 2022 + 1))

    # Clean and format the data
    emi_df2 = (
        # Rename columns
        emi_df2.rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        # Remove AK and HI
        .query("state_code not in ['AK', 'HI']")
        # Filter for specific animal category
        .query(f'animal.str.contains(r"{pattern2}", regex=True, na=False)',
               engine='python')
        .set_index(["state_code", "county", "fips", "month", "animal"])
        # Convert NA to 0 & Drop states with no data
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state/county/fips/month
        .melt(id_vars=["state_code", "county", "fips", "month", "animal"],
              var_name="year", value_name="ch4_tg")
        # Convert tg to kt
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # Ensure only years between min_year and max_year are included
        # Changed from max_year, since 2023 will be different
        .query("year.between(@min_year, 2022)")
        # Ensure state/county/fips/year/month grouping is unique
        # Fips kept in due to different counties having the same name
        .groupby(["state_code", "county", "fips", "year", "month", "animal"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
        .sort_values(by=['fips', 'year', 'month'])
        .reset_index()
        )

    # Look at 2022 proportions
    prop_2022 = (
        # Filter for 2022 data
        emi_df2.query("year == 2022")
        # Calculate proportions by state
        .assign(proportion=lambda x: x['ghgi_ch4_kt'] / x.groupby('state_code')['ghgi_ch4_kt'].transform('sum'))
        # Filter for V3/emi_file pattern
        .query(f'animal.str.contains(r"{pattern}", regex=True, na=False)', engine='python')
        # Group by fips to get total proportion by county
        .groupby(["state_code", "county", "fips", "year", "month"], as_index=False)["proportion"].sum()
    )
    # Check that proportions sum to 1 by state
    # prop_2022.groupby('state_code')['proportion'].sum()

    ################################################################################
    # Read in 2023 state level data and apply 2022 proportions to get county level data
    ################################################################################

    # Substrings for V4 pattern == V4 Category
    substrings3 = params["substrings3"]

    # Read in data
    emi_df3 = pd.read_excel(
        in_path[1],
        sheet_name=params["arguments"][2],  # Sheet name
    )

    emi_df3 = (
        # name column names lower
        emi_df3.rename(columns=lambda x: str(x).lower())
        # Rename state column
        .rename(columns={"georef": "state_code"})
        # Filter out national data
        .query("state_code not in ['AK', 'HI', 'AS', 'GU', 'MP', 'PR', 'UM', 'VI']")
        # Filter for Sinter Production & CH4
        .query("(subcategory1 == @substrings3) & (ghg == 'CH4')")
        # Filter for state_code and years
        .filter(items=["state_code", "2023"], axis=1)
        .set_index("state_code")
        # Replace NA values with 0
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
        # Ensure state/year grouping is unique
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
        )

    ################################################################################
    # Apply 2022 proportions to get county level data
    ################################################################################

    emi_df_2023 = (
        # Merge 2023 state-level data with 2022 proportions
        prop_2022
        .merge(emi_df3[['state_code', 'ghgi_ch4_kt']], on='state_code', how='left')
        # Disagreggate state 2023 data to county level using 2022 proportions
        .assign(
            ghgi_ch4_kt=lambda x: x['ghgi_ch4_kt'] * x['proportion'],
            year=2023
        )
        .drop(columns=['proportion'])
    )

    ################################################################################
    # Append 2023 data to 2012-2022 data
    ################################################################################

    # Append 2023 data to 2012-2022 data
    emi_df_final = pd.concat([emi_df, emi_df_2023], ignore_index=True)

    return emi_df_final


# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name = "3B_manure_management"
# Directory name for GHGI data
source_path = "3B_manure_management"

# Data Guide Directory
proxy_file_path = V4_DATA_PATH.parents[0] / "gch4i_data_guide_v4.xlsx"
# Read and query for the source name (ghch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_data_guide").query(
    f"gch4i_name == '{source_name}'"
)

# Initialize the emi_parameters_dict
emi_parameters_dict = {}
# Loop through the proxy data and store the parameters in the emi_parameters_dict
for emi_name, data in proxy_data.groupby("emi_id"):
    filenames = data.file_name.iloc[0].split(",")
    emi_parameters_dict[emi_name] = {
        "input_paths": [
            (v3_ghgi_data_dir_path if i == 0 else v4_ghgi_data_dir_path) / source_path /
            x for i, x in enumerate(filenames)
            ],
        "source_list": [x.strip().casefold() for x in data.Subcategory2.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": v4_emi_data_dir_path / f"{emi_name}.csv",
    }

emi_parameters_dict


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_livestock_manure_management_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for ghgi_group in source_list:
            individual_emi_df = get_livestock_manure_management_inv_data(
                input_paths,
                ghgi_group,
                parameters
            )
            emi_df_list.append(individual_emi_df)

        # Concatenate the emissions data and group by state and year
        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "county", "fips", "year", "month"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        # Save the emissions data to the output path
        emission_group_df.to_csv(output_path)

########################################################################################
# Testing
########################################################################################

# # %% Testing

# # # Read in emi outputs
# beef = pd.read_csv(v3_emi_data_dir_path / "manure_management_beef_emi.csv").query("state_code not in ['AK', 'HI']")
# bison = pd.read_csv(v3_emi_data_dir_path / "manure_management_bison_emi.csv").query("state_code not in ['AK', 'HI']")
# broilers = pd.read_csv(v3_emi_data_dir_path / "manure_management_broilers_emi.csv").query("state_code not in ['AK', 'HI']")
# chickens = pd.read_csv(v3_emi_data_dir_path / "manure_management_chickens_emi.csv").query("state_code not in ['AK', 'HI']")
# dairy = pd.read_csv(v3_emi_data_dir_path / "manure_management_dairy_emi.csv").query("state_code not in ['AK', 'HI']")
# goats = pd.read_csv(v3_emi_data_dir_path / "manure_management_goats_emi.csv").query("state_code not in ['AK', 'HI']")
# horses = pd.read_csv(v3_emi_data_dir_path / "manure_management_horses_emi.csv").query("state_code not in ['AK', 'HI']")
# layers = pd.read_csv(v3_emi_data_dir_path / "manure_management_layers_emi.csv").query("state_code not in ['AK', 'HI']")
# mules = pd.read_csv(v3_emi_data_dir_path / "manure_management_mules_emi.csv").query("state_code not in ['AK', 'HI']")
# pullets = pd.read_csv(v3_emi_data_dir_path / "manure_management_pullets_emi.csv").query("state_code not in ['AK', 'HI']")
# sheep = pd.read_csv(v3_emi_data_dir_path / "manure_management_sheep_emi.csv").query("state_code not in ['AK', 'HI']")
# swine = pd.read_csv(v3_emi_data_dir_path / "manure_management_swine_emi.csv").query("state_code not in ['AK', 'HI']")
# turkeys = pd.read_csv(v3_emi_data_dir_path / "manure_management_turkeys_emi.csv").query("state_code not in ['AK', 'HI']")
# cattle = pd.read_csv(v3_emi_data_dir_path / "manure_management_cattle_emi.csv").query("state_code not in ['AK', 'HI']")
# onfeed = pd.read_csv(v3_emi_data_dir_path / "manure_management_onfeed_emi.csv").query("state_code not in ['AK', 'HI']")

# # Quick check of year totals
# year_total_beef = beef.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_bison = bison.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_broilers = broilers.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_chickens = chickens.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_dairy = dairy.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_goats = goats.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_horses = horses.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_layers = layers.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_mules = mules.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_pullets = pullets.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_sheep = sheep.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_swine = swine.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_turkeys = turkeys.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_cattle = cattle.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_onfeed = onfeed.groupby("year")["ghgi_ch4_kt"].sum().reset_index()

# year_total_beef_joined = pd.concat([year_total_beef, year_total_cattle, year_total_onfeed], ignore_index=True).groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_poultry_joined = pd.concat([
#     year_total_broilers,
#     year_total_chickens,
#     year_total_layers,
#     year_total_pullets,
#     year_total_turkeys
#     ], ignore_index=True).groupby("year")["ghgi_ch4_kt"].sum().reset_index()

# # %% Build out Yearly totals from v4_source to compare with v3 yearly numbers
# year_list = [str(x) for x in list(range(min_year, max_year + 1))]
# # Read in data
# emi_test = (
#     pd.read_excel(
#         v4_ghgi_data_dir_path / "3B_manure_management" / "Manure_CH4-N2O_Output_1990-2023_State.xlsx",
#         sheet_name="InvDB")
#     .query("GeoRef not in ['AK', 'HI', 'AS', 'GU', 'MP', 'PR', 'UM', 'VI']")
#     .query("GHG == 'CH4'")
#     .rename(columns=lambda x: str(x).lower())
#     .rename(columns={"georef": "state_code"})
#     .filter(items=["subcategory1"] + year_list, axis=1)
#     .melt(id_vars=["subcategory1"], var_name="year", value_name="ch4_tg")
#     .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
#     .drop(columns=["ch4_tg"])
#     .astype({"year": int, "ghgi_ch4_kt": float})
#     .fillna({"ghgi_ch4_kt": 0})
#     .groupby(["subcategory1", "year"])["ghgi_ch4_kt"]
#     .sum()
#     .reset_index()
# )

# # Read in emi outputs
# beef_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_beef_emi.csv")
# bison_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_bison_emi.csv")
# broilers_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_broilers_emi.csv")
# chickens_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_chickens_emi.csv")
# dairy_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_dairy_emi.csv")
# goats_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_goats_emi.csv")
# horses_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_horses_emi.csv")
# layers_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_layers_emi.csv")
# mules_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_mules_emi.csv")
# pullets_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_pullets_emi.csv")
# sheep_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_sheep_emi.csv")
# swine_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_swine_emi.csv")
# turkeys_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_turkeys_emi.csv")
# cattle_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_cattle_emi.csv")
# onfeed_v4 = pd.read_csv(v4_emi_data_dir_path / "manure_management_onfeed_emi.csv")

# # Quick check of year totals
# year_total_beef_v4 = beef_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_bison_v4 = bison_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_broilers_v4 = broilers_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_chickens_v4 = chickens_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_dairy_v4 = dairy_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_goats_v4 = goats_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_horses_v4 = horses_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_layers_v4 = layers_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_mules_v4 = mules_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_pullets_v4 = pullets_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_sheep_v4 = sheep_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_swine_v4 = swine_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_turkeys_v4 = turkeys_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_cattle_v4 = cattle_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
# year_total_onfeed_v4 = onfeed_v4.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
