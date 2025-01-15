"""
Name:                   task_stat_comb_emi.py
Date Last Modified:     2024-12-12
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of mobile combustion emissions
Input Files:            - Stationary non-CO2 InvDB State Breakout_2022.xlsx
Output Files:           - Emissions by State, Year for each subcategory
Notes:
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

def get_comb_stat_inv_data(in_path, src, params):
    """
    Read in the ch4_kt values for each state.
    Output emissions files in kt for each state and year.

    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest
    params : dict
        additional parameters
    """

    # Read in the data
    emi_df = (
        pd.read_excel(
            in_path[0],
            sheet_name=params["arguments"][0],  # Sheet name
            skiprows=params["arguments"][1],  # Skip rows
            index_col=None
        )
    )
    # Initialize years
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    # Filter for columns
    emi_df = (
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"georef": "state_code"})
        # Query for CH4 emissions
        .query('ghg == "CH4"')
        # Filter for state_code, category, fuel type, and year columns
        .filter(items=["state_code", "category", "fuel1"] + year_list)
    )

    # Determine whether cateogry is Electricty Generation & Define fuel type
    # Filter for category and fuel type
    cat = src.split("_")[0].title()
    fuel = src.split("_")[1].split("--")[0].title()

    # Query for Electricity Generation & speicific fuel type, if category
    if cat == "Electricity Generation":
        emi_df = emi_df.query('category == "Electricity Generation" and fuel1 == @fuel')
    # Query for identified category and fuel type
    else:
        emi_df = emi_df.query('category == @cat and fuel1 == @fuel')

    # Clean the data
    emi_df = (
        # Drop columns after querying
        emi_df.drop(columns=["fuel1", "category"])
        .set_index("state_code")
        # Replace NA values with 0
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        # Convert tg to kt
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # Ensure state/year grouping is unique
        .groupby(["state_code", "year"]).sum().reset_index()
    )
    return emi_df

################################################################################
################################################################################

# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name = "1A_stationary_combustion"
# Directory name for GHGI data
source_path = "1A_stationary_combustion"  # Changed from combustion_stationary

# Data Guide Directory
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# Read and query for the source name (ghch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

# Edited for multiple filenames
# Initialize the emi_parameters_dict
emi_parameters_dict = {}
# Loop through the proxy data and store the parameters in the emi_parameters_dict
for emi_name, data in proxy_data.groupby("emi_id"):
    filenames = data.file_name.iloc[0].split(",")
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_path / x for x in filenames],
        "source_list": [x.strip().casefold() for x in (data.Category + "_" + data.Fuel1)
                        .to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict

# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_stat_comb_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for ghgi_group in source_list:
            individual_emi_df = get_comb_stat_inv_data(input_paths,
                                                       ghgi_group,
                                                       parameters)
            emi_df_list.append(individual_emi_df)

        # Concatenate the emissions data and group by state and year
        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        # Save the emissions data to the output path
        emission_group_df.to_csv(output_path)
