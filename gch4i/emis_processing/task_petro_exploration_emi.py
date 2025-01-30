"""
Name:                   task_petro_exploration.py
Date Last Modified:     2025-01-30
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of petroleum exploration systems emissions
                        to State, Year, emissions format
gch4i_name:             1B2ai_petroleum_exploration
Input Files:            {ghgi_data_dir_path}/1B2ai_petroleum_exploration/
                            Completions+Workovers_StateEstimates_2024.xlsx
                            Petro_Exploration_AllWell_OilWellDrilled_StateCH4.xlsx
Output Files:           - {emi_data_dir_path}/
                            oil_well_drilled_exp_emi.csv
                            oil_well_exp_emi.csv
                            pet_hf_comp_emi.csv
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

# %% STEP 1. Create Emi Mapping Functions


def get_petro_exploration_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    Subcategory Dictionary:
    well testing and non-hf completions
        - Non-completion well testing - Vented
        - Non-completion well testing - Flared
        - Well Completion Venting (less HF completions)
    Well Drilling
        - Well Drilling - Fugitive
        - Well Drilling - Combustion
    HF Completions - Total
        - HF Completions: Non-REC with Venting
        - HF Completions: Non-REC with Flaring
        - HF Completions: REC with Venting
        - HF Completions: REC with Flaring

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
    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Sheet name
        nrows=params["arguments"][1],  # Number of rows
        index_col=None
        )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .drop(columns="state")
        .rename(columns={"state code": "state_code"})
        .set_index("state_code")
        # Filter state code and years
        .filter(items=["state_code"] + year_list, axis=1)
        # Replace NA values with 0
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_mt")
        # Convert mt to kt
        .assign(ghgi_ch4_kt=lambda df: df["ch4_mt"] / 1000)
        .drop(columns=["ch4_mt"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # Ensure only years between min_year and max_year are included
        .query("year.between(@min_year, @max_year)")
        # Ensure state/year grouping is unique
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
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
source_name = "1B2ai_petroleum_exploration"
# Directory name for GHGI data
source_path = "1B2ai_petroleum_exploration"

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
    def task_ww_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_petro_exploration_inv_data(input_path,
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
