"""
Name:                   task_petro_production_emi.py
Date Last Modified:     2025-01-17
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of petroleum systems emissions
                        to State, Year, emissions format
gch4i_name:             1B2aii_petroleum_production
Input Files:            1B2aii_petroleum_production Input Files
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  -
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


# %% STEP 0.5 Create Function to Read Excel Parameters


def read_excel_params(file_path, subsector, emission, sheet='emi_proxy_mapping'):
    """
    Reads add_param column from gch4i_data_guide_v3.xlsx and returns a dictionary.
    """
    # Read in Excel File
    df = (pd.read_excel(file_path, sheet_name=sheet)
            .assign(
                ghgi_group=lambda x: x['Subcategory2'].str.strip().str.casefold()
            ))

    # Filter for Emissions Dictionary
    df = df.loc[df['gch4i_name'] == subsector]

    df = df.loc[df['ghgi_group'] == emission, 'add_params']

    # Convert to dictionary
    result = ast.literal_eval(df.iloc[0])

    return result


# %% STEP 1. Create Emi Mapping Functions


def get_petro_production_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    Subcategory Dictionary:
    Basin 360 - Total
        - Associated Gas Venting
        - Associated Gas Flaring
    Basin 220 - Total
        - Associated Gas Venting
        - Associated Gas Flaring
        - Miscellaneous Production Flaring
    Produced Water - Total
        - Produced Water - Regular Pressure Wells
        - Produced Water - Low Pressure Wells
    Basin 395 - Total
        - Associated Gas Venting
        - Associated Gas Flaring
        - Miscellaneous Production Flaring
    Basin 430 - Total
        - Associated Gas Venting
        - Associated Gas Flaring
        - Miscellaneous Production Flaring
    Basin Other - Total
        - Associated Gas Venting
        - Associated Gas Flaring
        - Miscellaneous Production Flaring
    Compressors, Blowdowns, Starts, Gas Engines
        - Compressor Blowdowns
        - Compressor Starts
        - Compressors
        - Gas Engines
    Non-HF Well Workovers
    Sales Areas, Heaters, Pressure Relief Valves
        - Sales Areas
        - Heaters
        - Pressure Relief Valves
    Well Blowouts Onshore
    Blowdowns, pipelines, battery pumps
        - Vessel Blowdowns
        - Pipelines
        - Battery Pumps
    Tanks
        - Large Tanks w/Flares
        - Small Tanks w/Flares
        - Large Tanks w/VRU
        - Large Tanks w/o Control
        - Small Tanks w/o Flares
        - Malfunctioning Separator Dump Valves
    wellheads, separators, headers, heaters
        - Oil Wellheads (heavy crude)
        - Oil Wellheads (light crude)
        - Separators (heavy crude)
        - Separators (light crude)
        - Heater/Treaters (light crude)
        - Headers (heavy crude)
        - Headers (light crude)
    Chemical Injection Pumps
    Pneumatic Devices - Total
        - Pneumatic Devices, High Bleed
        - Pneumatic Devices, Low Bleed
        - Pneumatic Devices, Int Bleed
    HF Workovers - Total
        - HF Workovers: Non-REC with Venting
        - HF Workovers: REC with Venting
    ----------

    This function takes different cleaning paths, depending on the emission source.
    Some emis group based on multiple sources, while others are based on a
    single source. The function reads in the data and returns the emissions in kt.

    ----------
    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest
    params : dict
        additional parameters
    """

    ####################################################################################

    # Overwrite parameters if Emi group is made up of mutliple srcs
    # If source is in this list, then params must be overwritten, as it contributes
    # to a combined emission group.
    if src in (['sales areas, heaters, pressure relief valves', 'tanks',
                'blowdowns, pipelines, battery pumps',
                'wellheads, separators, headers, heaters',
                'chemical injection pumps', 'pneumatic devices - total']):
        # Directly overwrite the params dictionary
        params = read_excel_params(proxy_file_path,
                                   source_name,
                                   src,
                                   sheet='emi_proxy_mapping')
    else:
        params = params

    # Read in the data
    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Sheet name
        nrows=params["arguments"][1],  # Number of rows
        skiprows=params["arguments"][2],  # Skip rows
        index_col=None
        )

    # Create a dictionary to map state names to state codes
    # Consider replacing this with pre-defined dictionary
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

    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

########################################################################################
    # Follow this path if the source is in the list
    if src in ["offshore gom federal waters",
               "offshore pacific federal and state waters",
               "offshore alaska state waters"]:
        # If True, 'make_up' list for queryign emission_source
        if src == "offshore alaska state waters":
            make_up = ["Offshore Alaska State Waters, Vent/Leak",
                       "Offshore Alaska State Waters, Flare"]
        # If True, 'make_up' list for queryign emission_source
        elif src == "offshore pacific federal and state waters":
            make_up = ["Offshore Pacific Federal and State Waters, Flare",
                       "Offshore Pacific Federal and State Waters, Vent/Leak"]
        # If True, 'make_up' list for queryign emission_source
        elif src == "offshore gom federal waters":
            make_up = ["Offshore GoM Federal Waters: Major Complexes",
                       "Offshore GoM Federal Waters: Minor Complexes",
                       "Offshore GoM Federal Waters: Flaring"]
        emi_df = (
            # Rename columns
            emi_df.rename(columns=lambda x: str(x).lower())
            # Filter for emission sources and years
            .filter(items=["emission source"] + year_list, axis=1)
            # Add '_' to emission source
            .rename(columns={"emission source": "emission_source"})
            # Query for 'make_up' in emissions_source
            .query("emission_source in @make_up")
            )

        # Transpose the dataframe
        emi_df = pd.DataFrame(emi_df.sum()).transpose()

        emi_df = (
            # Melt the data: unique state/year
            emi_df.melt(id_vars="emission_source", var_name="year", value_name="ch4_mt")
            # Convert mt to kt
            .assign(ghgi_ch4_kt=lambda x: x["ch4_mt"] / 1000)
            .drop(columns=["ch4_mt"])
            .astype({"year": int, "ghgi_ch4_kt": float})
            .fillna({"ghgi_ch4_kt": 0})
            # Ensure only years between min_year and max_year are included
            .query("year.between(@min_year, @max_year)")
            # Make state_code "ALL"
            .assign(state_code="ALL")
            # Ensure state/year grouping is unique
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        return emi_df

    ####################################################################################
    # Path Begins here, if not in previous list

    # 1. Offshore Alaska State Waters, Vent/Leak
    # 2. Offshore Alaska State Waters, Flare

    # 1. Offshore Pacific Federal and State Waters, Flare
    # 2. Offshore Pacific Federal and State Waters, Vent/Leak

    # 1. Offshore GoM Federal Waters: Major Complexes (there are THREE instances of this source, which need to be summed)
    # 2. Offshore GoM Federal Waters: Minor Complexes (there are THREE instances of this source, which need to be summed)
    # 3. Offshore GoM Federal Waters: Flaring

    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .assign(state_code=lambda x: x["state"].str.lower().map(state_dict))
        .filter(items=["state_code"] + year_list, axis=1)
        .set_index("state_code")
        # Replace NA values with 0
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
    )
    # If-else statement: if source is in the list, then follow this path
    if src in ["tanks", "wellheads, separators, headers, heaters"]:
        emi_df = (
            # Melt the data: unique state/year
            emi_df.melt(id_vars="state_code", var_name="year", value_name="ch4_mt")
            # Convert mt to kt
            .assign(ghgi_ch4_kt=lambda x: x["ch4_mt"] / 1000)
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
    # Else Path
    else:
        emi_df = (
            # Melt the data: unique state/year
            emi_df.melt(id_vars="state_code", var_name="year", value_name="ghgi_ch4_kt")
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
source_name = "1B2aii_petroleum_production"
# Directory name for GHGI data
source_path = "1B2aii_petroleum_production"

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
    def task_petro_prod_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_petro_production_inv_data(input_path,
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
