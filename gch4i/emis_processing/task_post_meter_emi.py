"""
Name:                   task_post_meter_emi.py
Date Last Modified:     2024-12-16
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of post_meter emissions to State, Year, emissions format
gch4i_name:             1B2bvi1_ng_postmeter
Input Files:            - {ghgi_data_dir_path}/{1B2bvi1_ng_postmeter}/
                            Emi_CommCustomers.xlsx
                            Emi_ResCustomers.xlsx
                            Emi_CNGVehicles.xlsx
                            Emi_IndEGU.xlsx
Output Files:           - {emi_data_dir_path}/
                            postmeter_comm_emi.csv
                            postmeter_resi_emi.csv
                            postmeter_vehicle_emi.csv
                            postmeter_ind_egu_emi.csv
"""
# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
)


# %% Step 1. Create Function


def get_postmeter_inv_data(in_path, src):
    """
    read in the ghgi_ch4_kt values for each state

    NOTE: Residential (NET): These emissions are the net of residential emissions,
        minus the deductions from the energy chapter
    NOTE: Industrials + EGUs: This source includes the following subcategories:
        - Industrials
        - EGUs

    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest
    params : dict
        additional parameters
    """

    # Create a dictionary to map state names to state codes
    # Consider replacing with us_state_to_abbrev mapping
    state_dict = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
        "district of columbia": "DC",
        "puerto rico": "PR",
        "guam": "GU",
        "u.s. minor outlying islands": "UM",
        "u.s. virgin islands": "VI",
        "virgin islands": "VI",
        "american samoa": "AS",
        "northern mariana islands": "MP",
        "northern mariana is": "MP"
    }

    # Read in the data
    emi_df = pd.read_excel(
        in_path,
        sheet_name="Sheet1",
        skiprows=0,
        nrows=52,
    )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        # Map state names to state codes
        .assign(state_code=lambda x: x["state"].str.lower().map(state_dict))
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
        .melt(id_vars="state_code", var_name="year", value_name="ghgi_ch4_kt")
        .astype({"year": "int", "ghgi_ch4_kt": "float"})
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
source_name = "1B2bvi1_ng_postmeter"
# Directory name for GHGI data
source_path = "1B2bvi1_ng_postmeter"

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
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_petro_prod_emi(
        input_paths: list[Path],
        source_list: list[str],
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_postmeter_inv_data(input_path,
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
