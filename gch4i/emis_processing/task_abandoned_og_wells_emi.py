"""
Name:                   task_abandoned_og_wells_emi.py
Date Last Modified:     2025-06-10
Authors Name:           A. Burnette, Nick Kruskamp (RTI International)
Purpose:                Mapping of wells emissions to State, Year, emissions format
gch4i_name:             1B2ab_abandoned_og_wells
Input Files:            - {ghgi_data_dir_path}/1B2ab_abandoned_og_wells/
                            Abandoned_Wells_90-22_FR.xlsx
Output Files:           - {emi_data_dir_path}/
                            aog_gas_wells_emi.csv
                            aog_oil_wells_emi.csv
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


def get_abandoned_og_wells_inv_data(in_path, src, params):
    """
    Change from previous years: no Plugged or Regional emis. All emissions are either
    from "gas" or "oil" wells.

    Function reads in the inventory data for abandoned oil and gas wells and returns
    the emissions in kt for each state and year.

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
        sheet_name=params["arguments"][0],  # sheet name
        skiprows=params["arguments"][1],  # skip rows
        nrows=params["arguments"][2],  # number of rows
    )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .assign(
            ghgi_source=lambda df: df["subcategory2"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        .rename(columns={"georef": "state_code"})
        # Query for CH4 emissions and the source of interest
        .query(f"(ghg == 'CH4') & (ghgi_source == '{src}')")
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


# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name = "1B2ab_abandoned_og_wells"
# Directory name for GHGI data
source_path = "1B2ab_abandoned_og_wells"  # Changed from abandoned_og_wells

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
    print(data)
    emi_parameters_dict[emi_name] = {
        "input_path": ghgi_data_dir_path / source_path / data.file_name.values[0],
        "emi_source": data.Subcategory2.str.strip().str.casefold().values[0],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
    }

emi_parameters_dict
# %%
input_path, emi_source, parameters, output_path = emi_parameters_dict[
    "aog_gas_wells_emi"
].values()
input_path, emi_source, parameters, output_path
# %%
emi_df = get_abandoned_og_wells_inv_data(input_path, emi_source, parameters)
emi_df.query("state_code == 'ID'").head(10)
emi_df.query("state_code == 'ID'").head(10)
emi_df.query("state_code == 'ID'").head(10)
# %% STEP 3. Create Pytask Function and Loop


def task_abandoned_og_wells_emi(
    input_path: Path,
    emi_source: str,
    parameters: dict,
    output_path: Annotated[Path, Product],
) -> None:

    emi_df = get_abandoned_og_wells_inv_data(input_path, emi_source, parameters)
    # Save the emissions data to the output path
    emi_df.to_csv(output_path)


# %%
import pytask

sesh = pytask.build(
    tasks=[
        task_abandoned_og_wells_emi(**kwargs)
        for _id, kwargs in emi_parameters_dict.items()
    ]
)
sesh

# %%
