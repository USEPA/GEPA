"""
Name:                   task_wetlands_rem_wet_emi.py
Date Last Modified:     2025-06-13
Authors Name:           Andrew Burnette, Nick Kruskamp (RTI International)
Purpose:                Mapping of wetlands remaining wetlands emissions to State, Year,
                            emissions format
gch4i_name:             4D1_wetlands_remaining_wetlands
Input Files:            - {ghgi_data_dir_path}/4D1_wetlands_remaining_wetlands/
                            FloodedLands_90-22_State.xlsx [InvDB]
                            Peatlands_90-22_State_ERG_05.14.24.xlsx [InvDB]
                            CoastalWetlands_90-22_FR.xlsx [InvDB]
Emis/Output Files:      - {emi_data_dir_path}/
                            rem_flooded_land_reservoir_emi.csv
                            rem_flooded_land_other_emi.csv
                            peatlands_emi.csv
                            rem_coastal_wetlands_emi.csv
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
    min_year,
)
from gch4i.utils import tg_to_kt

# %% Step 1. Create Function


def get_wetlands_rem_wet_inv_data(in_path, params):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - Flooded Land Remaining Flooded Land: Reservoir
    - Flooded Land Remaining Flooded Land: other constructed waterbodies
    - Wetlands Remaining Wetlands: Peatlands
    - Wetlands Remaining Wetlands: Coastal Wetlands Remaining Coastal Wetlands

    Parameters
    ----------
    in_path : str
        path to the input file
    params : dict
        additional parameters
    """

    # Read in the data
    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Sheet name
        skiprows=params["arguments"][1],  # Skip rows
    )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    # Create state_list to filter states
    state_list = emi_df["GeoRef"].unique().tolist()
    state_list = [
        state
        for state in state_list
        if state not in ["AS", "GU", "MP", "PR", "VI", "AK", "HI", "National"]
    ]

    cat, subcat_1, subcat_2 = params['substrings']

    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .assign(
            ghgi_source=lambda df: df["subcategory1"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        .rename(columns={"georef": "state_code"})
        # Query for states in state_list
        .query("state_code in @state_list")
        # Query for CH4 emissions and the source of interest
        .query(f"(ghg == 'CH4')")
        # Query for the subcategory combinations needed
        .query(
            f"(category == '{cat}')"
            f"& (subcategory1 == '{subcat_1}')"
            # f"& (subcategory2.isin({subcat_2}))",
        )
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
    emi_df

    return emi_df


# %% STEP 2. Initialize Paradmeters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name_1 = "4D1_wetlands_remaining_wetlands"
source_name_2 = "4D2_land_converted_to_wetlands"
# Data Guide Directory
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# Read and query for the source name (ghch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"(gch4i_name == '{source_name_1}') | (gch4i_name == '{source_name_2}')"
)

# Initialize the emi_parameters_dict
emi_parameters_dict = {}
# Loop through the proxy data and store the parameters in the emi_parameters_dict
for emi_name, data in proxy_data.groupby("emi_id"):
    print(data)
    emi_parameters_dict[emi_name] = {
        "input_path": ghgi_data_dir_path
        / data.gch4i_name.values[0]
        / data.file_name.values[0],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
    }

emi_parameters_dict
# %% STEP 3. Create Pytask Function and Loop

def task_flooded_lands_emis(
    input_path: Path,
    parameters: dict,
    output_path: Annotated[Path, Product],
) -> None:

    emi_df = get_wetlands_rem_wet_inv_data(input_path, parameters)
    # Save the emissions data to the output path
    emi_df.to_csv(output_path)


# %%
import pytask

sesh = pytask.build(
    tasks=[
        task_flooded_lands_emis(**kwargs)
        for _id, kwargs in emi_parameters_dict.items()
    ]
)
sesh

# %%
