"""
Name:                   task_iron_steel_emi.py
Date Last Modified:     2025-02-13
Authors Name:           Chris Coxen
Purpose:                Mapping of iron and steel emissions to State, Year, emissions
                        format
gch4i_name:             2C1_iron_and_steel
Input Files:            - {ghgi_data_dir_path}/2C1_iron_and_steel/
                            State_Iron-Steel_1990-2022.xlsx
Output Files:           - {emi_data_dir_path}/
                            iron_steel_emi.csv
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    min_year,
    max_year
)
from gch4i.utils import tg_to_kt


# %% Step 1. Create Function


def get_iron_and_steel_inv_data(in_path, src):
    """read in the ch4_kt values for each state

    Function reads in the inventory data for iron and steel and returns the
    emissions in kt for each state and year.

    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest

    Returns
        Saves the emissions data to the output path.
    """

    # Read in the data
    emi_df = pd.read_excel(
        in_path,
        sheet_name="InvDB",
        skiprows=15,
        nrows=457,
        usecols="A:BA"
        )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    # Clean and format the data
    emi_df = (
        # name column names lower
        emi_df.rename(columns=lambda x: str(x).lower())
        # Rename state column
        .rename(columns={"georef": "state_code"})
        # Filter out national data
        .query("state_code != 'National'")
        # Filter for Sinter Production & CH4
        .query("(subcategory1 == 'Sinter Production') & (ghg == 'CH4')")
        # Filter for state_code and years
        .filter(items=["state_code"] + year_list, axis=1)
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
    return emi_df


# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name = "2C1_iron_and_steel"
# Directory name for GHGI data
source_path = "2C1_iron_and_steel"

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
        "source_list": [x.strip().casefold() for x in data.Category.to_list()],
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_iron_steel_emi(
        input_paths: list[Path],
        source_list: list[str],
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_iron_and_steel_inv_data(input_path,
                                                            ghgi_group)
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
