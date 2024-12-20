"""
Name:                   task_livestock_manure_management.py
Date Last Modified:     2024-10-11
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of Livestock emissions to Year, Month, State, County,
                        emissions format
Input Files:            - Monthly_Manure_Output (Temp file goes to 2018)
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
from gch4i.utils import tg_to_kt

# %% Step 1. Create Function


def get_livestock_manure_management_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - Beef
        - beef_NOF_bull
        - beef_NOF_cow
        - calf_NOF_beef
    - Cattle
        - beef_NOF_steer
        - beef_NOF_heifers
    - Dairy
        - dairy_cow
        - dairy_heifers
        - calf_NOF_dairy
    - OnFeed
        - beef_OF_heifers
        - beef_OF_steer
    - Bison
    - Goats
    - Horses
    - Mules
    - Sheep
    - Swine
        - swine_50
        - swine_50_119
        - swine_120_179
        - swine_180
        - swine_breeding
    - Broilers
        - poultry_broilers
    - Layers
        - poultry_layers
    - Turkeys
        - poultry_turkeys
    - Chickens
        - poultry_chickens
    - Pullets
        - poultry_pullets

    """

    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Gridded_Methane___Enteric_total
        skiprows=params["arguments"][1],  # 2
    )

    # Fix year column names
    emi_df = emi_df.drop(emi_df.columns[0], axis=1)
    emi_df.columns.values[5:] = list(range(min_year, max_year + 1))

    # Format data
    emi_df = (
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        .query(f'animal.str.contains("{params["substrings"][0]}", regex=True)',
               engine='python')  # param
        .drop(columns=['animal'])
        .set_index(["state_code", "county", "fips", "month"])
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        .melt(id_vars=["state_code", "county", "fips", "month"],
              var_name="year", value_name="ch4_tg")
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        .query("year.between(@min_year, @max_year)")
        .groupby(["state_code", "county", "fips", "year", "month"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
        .sort_values(by=['fips', 'year', 'month'])
        .reset_index()
        )

    return emi_df


# %% STEP 2. Initialize Parameters
source_name = "3B_manure_management"
source_path = "3B_manure_management"

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
    def task_livestock_manure_management_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_livestock_manure_management_inv_data(input_path,
                                                                         ghgi_group,
                                                                         parameters)
            emi_df_list.append(individual_emi_df)

        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "county", "fips", "year", "month"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        emission_group_df.to_csv(output_path)
