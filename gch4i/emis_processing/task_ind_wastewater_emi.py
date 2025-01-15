"""
Name:                   task_ind_wastewater_emi.py
Date Last Modified:     2024-08-27
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of coal emissions to State, Year, emissions format
Input Files:            - WW_State-level Estimates_90-22_27June2024.xlsx
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
    ghgi_data_dir_path,
    max_year,
    min_year
)


# %% STEP 1. Create Emi Mapping Functions

def get_ind_ww_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - Breweries
    - Ethanol Production
    - Fruits and Vegetables
    - Red Meat and Poultry
    - Petroleum Refining
    - Pulp and Paper
    """

    emi_df = (
        # read in the data
        pd.read_excel(
            in_path,
            sheet_name=params["arguments"][0],  # Sheet name
            skiprows=params["arguments"][1],  # skip rows
            nrows=params["arguments"][2],  # number of rows
        )
    )
    # Will need an if-else statement for ww_mp_emi. Adjust for no column headers
    if src == "red meat and poultry":
        emi_df.drop(emi_df.columns[0:3], axis=1, inplace=True)
        emi_df.columns.values[0] = "State"
        emi_df.columns.values[1] = "Metric"
        emi_df.columns.values[2:] = list(range(1990, max_year + 1))
        emi_df = emi_df.drop(columns="Metric")

    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    emi_df = (
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        .filter(items=["state_code"] + year_list, axis=1)
        .set_index("state_code")
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        .melt(id_vars="state_code", var_name="year", value_name="ch4_metric")
        .assign(ghgi_ch4_kt=lambda df: df["ch4_metric"]/params["arguments"][3])
        .drop(columns=["ch4_metric"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        .query("year.between(@min_year, @max_year)")
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
    )
    return emi_df


# %% STEP 2. Initialize Parameters
source_name = "5D2_industrial_wastewater"
source_path = "wastewater"

proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

emi_parameters_dict = {}
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
    def task_ww_ind_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_ind_ww_inv_data(input_path, ghgi_group, parameters)
            emi_df_list.append(individual_emi_df)

        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        emission_group_df.to_csv(output_path)
