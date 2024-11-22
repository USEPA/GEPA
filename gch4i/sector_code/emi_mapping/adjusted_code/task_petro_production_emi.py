"""
Name:                   task_petro_production_emi.py
Date Last Modified:     2024-08-27
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of petroleum systems emissions
                        to State, Year, emissions format
Input Files:            1B2aii_petroleum_production.xlsx
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  - This version of emi mapping is draft for mapping .py files
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
    tmp_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
)
from gch4i.sector_code.emi_mapping.a_excel_dict import read_excel_params2

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
    """

    if src in (['sales areas, heaters, pressure relief valves', 'tanks',
                'blowdowns, pipelines, battery pumps',
                'wellheads, separators, headers, heaters',
                'chemical injection pumps', 'pneumatic devices - total']):
        params = read_excel_params2(proxy_file_path,
                                    source_name,
                                    src,
                                    sheet='emi_proxy_mapping')
    else:
        params = params

    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Sheet name
        nrows=params["arguments"][1],  # number of rows
        skiprows=params["arguments"][2],  # number of rows to skip
        index_col=None
        )

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

    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

########################################################################################
    # Edits happened here
    if src in ["offshore gom federal waters",
               "offshore pacific federal and state waters",
               "offshore alaska state waters"]:
        if src == "offshore alaska state waters":
            make_up = ["Offshore Alaska State Waters, Vent/Leak",
                       "Offshore Alaska State Waters, Flare"]
        elif src == "offshore pacific federal and state waters":
            make_up = ["Offshore Pacific Federal and State Waters, Flare",
                       "Offshore Pacific Federal and State Waters, Vent/Leak"]
        elif src == "offshore gom federal waters":
            make_up = ["Offshore GoM Federal Waters: Major Complexes",
                       "Offshore GoM Federal Waters: Minor Complexes",
                       "Offshore GoM Federal Waters: Flaring"]
        emi_df = (
            emi_df.rename(columns=lambda x: str(x).lower())
            .filter(items=["emission source"] + year_list, axis=1)
            .rename(columns={"emission source": "emission_source"})
            .query("emission_source in @make_up")
            )

        emi_df = pd.DataFrame(emi_df.sum()).transpose()

        emi_df = (
            emi_df.melt(id_vars="emission_source", var_name="year", value_name="ch4_mt")
            .assign(ghgi_ch4_kt=lambda x: x["ch4_mt"] / 1000)
            .drop(columns=["ch4_mt"])
            .astype({"year": int, "ghgi_ch4_kt": float})
            .fillna({"ghgi_ch4_kt": 0})
            .query("year.between(@min_year, @max_year)")
            .assign(state_code="ALL")
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        return emi_df

# 1. Offshore Alaska State Waters, Vent/Leak
# 2. Offshore Alaska State Waters, Flare

# 1. Offshore Pacific Federal and State Waters, Flare
# 2. Offshore Pacific Federal and State Waters, Vent/Leak

# 1. Offshore GoM Federal Waters: Major Complexes (there are THREE instances of this source, which need to be summed)
# 2. Offshore GoM Federal Waters: Minor Complexes (there are THREE instances of this source, which need to be summed)
# 3. Offshore GoM Federal Waters: Flaring

# To here
##########################
    emi_df = (
        emi_df.rename(columns=lambda x: str(x).lower())
        .assign(state_code=lambda x: x["state"].str.lower().map(state_dict))
        .filter(items=["state_code"] + year_list, axis=1)
        .set_index("state_code")
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
    )
    if src in ["tanks", "wellheads, separators, headers, heaters"]:
        emi_df = (
            emi_df.melt(id_vars="state_code", var_name="year", value_name="ch4_mt")
            .assign(ghgi_ch4_kt=lambda x: x["ch4_mt"] / 1000)
            .drop(columns=["ch4_mt"])
            .astype({"year": int, "ghgi_ch4_kt": float})
            .fillna({"ghgi_ch4_kt": 0})
            .query("year.between(@min_year, @max_year)")
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
    else:
        emi_df = (
            emi_df.melt(id_vars="state_code", var_name="year", value_name="ghgi_ch4_kt")
            .astype({"year": int, "ghgi_ch4_kt": float})
            .fillna({"ghgi_ch4_kt": 0})
            .query("year.between(@min_year, @max_year)")
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
    return emi_df


# %% STEP 2. Initialize Parameters
source_name = "1B2aii_petroleum_production"
source_path = "Petroleum and Natural Gas"

proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

emi_parameters_dict = {}
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

        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_petro_production_inv_data(input_path,
                                                              ghgi_group,
                                                              parameters)
            emi_df_list.append(individual_emi_df)

        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        emission_group_df.to_csv(output_path)


##########################################################################
# %% TESTING

# emi_df = pd.read_excel(
#     "/Users/aburnette/Library/CloudStorage/OneDrive-SharedLibraries-EnvironmentalProtectionAgency(EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/ghgi/Petroleum and Natural Gas/PetroleumSystems_90-22_FR.xlsx",
#     sheet_name="Production_CH4 (MT)",  # Sheet name
#     nrows=130,  # number of rows
#     skiprows=3,  # number of rows to skip
#     index_col=None
#     )

# make_up = ["Offshore Alaska State Waters, Vent/Leak",
#            "Offshore Alaska State Waters, Flare"]

# year_list = [str(x) for x in list(range(min_year, max_year + 1))]

# emi_df = (
#     emi_df.rename(columns=lambda x: str(x).lower())
#     .filter(items=["emission source"] + year_list, axis=1)
#     .rename(columns={"emission source": "emission_source"})
#     .query("emission_source in @make_up")
#     )

# emi_df = pd.DataFrame(emi_df.sum()).transpose()

# emi_df = (
#     emi_df.melt(id_vars="emission_source", var_name="year", value_name="ch4_mt")
#     .assign(ghgi_ch4_kt=lambda x: x["ch4_mt"] / 1000)
#     .drop(columns=["ch4_mt"])
#     .astype({"year": int, "ghgi_ch4_kt": float})
#     .fillna({"ghgi_ch4_kt": 0})
#     .query("year.between(@min_year, @max_year)")
#     .assign(state_code="ALL")
#     .groupby(["state_code", "year"])["ghgi_ch4_kt"]
#     .sum()
#     .reset_index()
# )



# # divide by 1000 to convert to kt
