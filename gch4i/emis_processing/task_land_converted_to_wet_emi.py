"""
Name:                   task_land_converted_to_wet_emi.py
Date Last Modified:     2025-01-30
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of land converted to wetlands emissions to State, Year,
                            and emissions format
gch4i_name:             4D2_land_converted_to_wetlands
Input Files:            - {ghgi_data_dir_path}/4D2_land_converted_to_wetlands/
                            FloodedLands_90-22_State.xlsx [InvDB]
                            CoastalWetlands_90-22_FR.xlsx [InvDB]
Emis/Output Files:      - {emi_data_dir_path}/
                            conv_flooded_land_reservoir_emi.csv
                            conv_coastal_wetlands_emi.csv
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


def get_converted_wetlands_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - Land Converted to Flooded Land
    - Land Converted to Wetlands

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
        skiprows=params["arguments"][1],  # Skip rows
        # nrows=params["arguments"][2],  # Number of rows
        )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    # Filter out non-CONUS states, territories, and national values
    state_list = emi_df["GeoRef"].unique().tolist()
    state_list = [state for state in state_list if state not in ["AS", "GU", "MP", "PR",
                                                                 "VI", "AK", "HI",
                                                                 "National"]]

    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .assign(
            ghgi_source=lambda df: df["subcategory1"]
            .astype(str)
            .str.strip()
            .str.casefold(),
            subcategory2=lambda df: df["subcategory2"]
            .astype(str)
            .str.strip()
        )
        .rename(columns={"georef": "state_code"})
        # Query for CH4 emissions, CONUS states, and the source of interest
        .query("state_code in @state_list")
        .query(f"(ghg == 'CH4') & (ghgi_source == '{src}')")
        # Query for the subcategories of interest (several layers per source)
        .query(f"(category == '{params['substrings'][0]}') & (subcategory1 == '{params['substrings'][1]}') & (subcategory2.isin({params['substrings'][2]}))", engine="python")
        # Filter state code and years
        .filter(items=["state_code"] + year_list, axis=1)
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
        # Ensure only years between min_year and max_year are included
        .query("year.between(@min_year, @max_year)")
        # Ensure state/year grouping is unique
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
        )
    return emi_df


# %% STEP 2. Initialize Paradmeters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name = "4D2_land_converted_to_wetlands"
# Directory name for GHGI data
source_path = "4D2_land_converted_to_wetlands"

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
        "source_list": [x.strip().casefold() for x in data.gch4i_source.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_land_convert_wetlands_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_converted_wetlands_inv_data(input_path,
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
