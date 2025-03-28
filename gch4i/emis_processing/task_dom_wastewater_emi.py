"""
Name:                   task_dom_wastewater_emi.py
Date Last Modified:     2025-01-30
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of domestic wastewater emissions to State, Year,
                        emissions format
gch4i_name:             5D1_domestic_wastewater
Input Files:            - {ghgi_data_dir_path}/5D1_domestic_wastewater/
                            WW_State-level Estimates_90-22_27June2024.xlsx
Output Files:           - {emi_data_dir_path}/
                            ww_dom_nonseptic_emi.csv
                            ww_sep_emi.csv
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import ast

from gch4i.config import (  # noqa
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
)

# %% STEP 1. Create Emi Mapping Functions


def get_dom_ww_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    Subcategory Dictionary:
    - Domestic - non-septic
        -Treated Aerobic Systems (Gg) - B
        - Emissions from Centrally Treated Anaerobic Systems (Gg) - C
        - Emissions from Anaerobic Sludge Digesters (D)
        - Centrally-Treated Wastewater Effluent (E)
    - Emissions from Septic Systems (Gg) - A

    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest
    params : dict
        additional parameters
    """

    emi_df = (
        pd.read_excel(
            in_path,
            sheet_name=params["arguments"][0],  # Sheet name
            skiprows=params["arguments"][1],  # skip rows
            nrows=params["arguments"][2],  # number of rows
        )
    )
    # Clean and select necessary columns
    emi_df = emi_df.drop(emi_df.columns[0:3], axis=1)
    emi_df.columns.values[0] = "State"
    emi_df.columns.values[1:] = list(range(1990, max_year + 1))

    # Clean and format data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        .set_index("state_code")
        # Replace NA values with 0
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state_code, year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_gg")
        # Change gg to kt (same metric unit)
        .rename(columns={"ch4_gg": "ghgi_ch4_kt"})
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
source_name = "5D1_domestic_wastewater"
# Directory name for GHGI data
source_path = "5D1_domestic_wastewater"  # Changed from wastewater

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
        "source_list": [x.strip().casefold() for x in data.Subcategory1.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
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
            individual_emi_df = get_dom_ww_inv_data(input_path, ghgi_group, parameters)
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
