"""
Name:                   task_livestock_manure_management.py
Date Last Modified:     2024-12-3
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of Livestock emissions to Year, Month, State, County,
                        emissions format
gch4i_name:             3B_manure_management
Input Files:            - {ghgi_data_dir_path}/3B_manure_management/
                            Gridded Methane - Manure emissions by County_v1_17Sept2024
                            .xlsx
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
"""
# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import ast

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
)
from gch4i.utils import tg_to_kt

# %% Step 1. Create Function


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
        in_path,
        sheet_name=params["arguments"][0],  # Sheet Name
        skiprows=params["arguments"][1],  # Skip Rows
    )

    # Fix year column names
    emi_df = emi_df.drop(emi_df.columns[0], axis=1)
    emi_df.columns.values[5:] = list(range(min_year, max_year + 1))

    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        # Filter for specific animal category
        .query(f'animal.str.contains("{params["substrings"][0]}", regex=True)',
               engine='python')  # param
        .drop(columns=['animal'])
        .set_index(["state_code", "county", "fips", "month"])
        # Convert NA values to 0 & Drop states with no data
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state/county/fips/month
        .melt(id_vars=["state_code", "county", "fips", "month"],
              var_name="year", value_name="ch4_tg")
        # Convert tg to kt
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # Ensure only years between min_year and max_year are included
        .query("year.between(@min_year, @max_year)")
        # Ensure state/county/fips/year/month grouping is unique
        # Fips kept in due to different counties having the same name
        .groupby(["state_code", "county", "fips", "year", "month"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
        .sort_values(by=['fips', 'year', 'month'])
        .reset_index()
        )

    return emi_df


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
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# Read and query for the source name (ghch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

# Initialize the emi_parameters_dict
emi_parameters_dict = {}
# Loop through the proxy data and store the parameters in the emi_parameters_dict
for emi_name, data in proxy_data.groupby("emi_id"):
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_path / x for x in data.file_name],
        "source_list": [x.strip().casefold() for x in data.Subcategory2.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
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
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_livestock_manure_management_inv_data(input_path,
                                                                         ghgi_group,
                                                                         parameters)
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
