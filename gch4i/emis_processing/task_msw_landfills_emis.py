"""
Name:                   task_msw_landfills_emis.py
Date Last Modified:     2024-12-16
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of msw_landfill emissions to State, Year, emissions
                        format
gch4i_name:             5A_msw_landfills
Input Files:            - {ghgi_data_dir_path}/{5A_msw_landfills}/
                            State_MSW_LF_1990-2022_LA.xlsx
Output Files:           - {emi_data_dir_path}/
                            msw_landfills_r_emi.csv
                            msw_landfills_nr_emi.csv
Notes:                  - This version of task_msw_landfills_emis.py is an updated
                        version of the non-data_guide pytask file
"""
# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated
from pytask import Product, task, mark

import geopandas as gpd
import pandas as pd
import ast

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
    )

# from gch4i.utils import tg_to_kt
tg_to_kt = 1000

state_path = Path(global_data_dir_path) / "tl_2020_us_state.zip"

# %% Step 1. Create Function


def get_msw_landfills_inv_data(in_path, src, params):
    """
    read in the ghgi_ch4_kt values for each state.

    Non-reporting emissions are calculated by scaling reporting emissions:
        Assume emissions are 9% of reporting emissions for 2016 and earlier
        Assume emissiosn are 11% of reporting emissions for 2017 and later

    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest
    params : dict
        additional parameters
    """
    # Get state vectors and state_code for use with inventory and proxy data
    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    # Read in the data
    emi_df = pd.read_excel(
        in_path,
        sheet_name="InvDB",
        skiprows=15,
        nrows=58,
        )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
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
        # Query for CH4 emissions and the source of interest
        .query(f"(ghg == 'CH4') & (ghgi_source == '{src}')")
        # Query for lower 48 + DC state_codes
        .query("state_code in @state_gdf.state_code")
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

    # Calculate Reporting vs Non-reporting emissions
    # Initiliaze non-reporting emissions
   
    nonreporting_emi_df = pd.DataFrame()
    # Scale non-reporting emissions
    emi_09 = (
        emi_df
        .query("year <= 2016")
        .assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * 0.09) #these factors follow the approach used in the GHGI
        )
    emi_11 = (
        emi_df
        .query("year >= 2017")
        .assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * 0.11) #these factors follow the approach used in the GHGI
        )
    # Concatenate non-reporting emissions
    nonreporting_emi_df = pd.concat([nonreporting_emi_df, emi_09, emi_11],
                                    axis=0)
    # Calculate reporting emissions by subtracting non-reporting emissions from the base
    reporting_emi_df = (
        pd.merge(
            emi_df,
            nonreporting_emi_df,
            on=["state_code", "year"],
            how="left",
            suffixes=("_base", "_nr")
        )
        .assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt_base"] - df["ghgi_ch4_kt_nr"])
        .drop(columns=["ghgi_ch4_kt_base", "ghgi_ch4_kt_nr"])
    )

    # Return the emissions data
    if params["arguments"][0] == "non-reporting":
        return nonreporting_emi_df
    elif params["arguments"][0] == "reporting":
        return reporting_emi_df
    else:
        print("Invalid argument. Please check data_guide parameters.")


# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name = "5A_msw_landfills"
# Directory name for GHGI data
source_path = "5A_msw_landfills"

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
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_msw_landfills_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_msw_landfills_inv_data(input_path,
                                                           ghgi_group,
                                                           parameters)
            emi_df_list.append(individual_emi_df)

        # Concatenate the emissions data and group by state and year
        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .query("ghgi_ch4_kt > 0")
            .reset_index()
        )
        # Save the emissions data to the output path
        emission_group_df.to_csv(output_path)
