"""
Name:                   task_petro_systems.py
Date Last Modified:     2024-08-19
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of petroleum systems emissions
                        to State, Year, emissions format
Input Files:            - 3 files, each emission has its own tab
                        - ChemicalInjectionPumps_StateEstimates_2024.xlsx,
                            Petro_State_CH4_MT
                        - PneumaticControllers_StateEstimates_2024.xlsx,
                            Petro_State_CH4_MT
                        - Completions+Workovers_StateEstimates_2024.xlsx,
                            Petro_State_HFComp_CH4
                            Petro_State_HFWOs_CH4
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  - This version of emi mapping is draft for mapping .py files
"""

# WARNING: Chemical Injection Pumps and Pneumatic Devices - Total are oil_well_prod_emis
# and may be part of a different emi mapping function.


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

from gch4i.sector_code.emi_mapping.a_excel_dict import read_excel_params

# %% STEP 1. Create Emi Mapping Functions


def get_petro_systems_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - chemical injection pumps
    - pneumatic devices
    - hf completions
    - hf workovers

    """

    if src in ["chemical injection pumps", "pneumatic devices"]:
        params = read_excel_params(proxy_file_path, subsector=source_name, emission=src, sheet='testing')
    else:
        params = params

    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Sheet name
        nrows=params["arguments"][1],  # number of rows
        index_col=None
        )

    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    emi_df = (
        emi_df.rename(columns=lambda x: str(x).lower())
        .drop(columns="state")
        .rename(columns={"state code": "state_code"})
        .set_index("state_code")
        .filter(items=["state_code"] + year_list, axis=1)
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        .melt(id_vars="state_code", var_name="year", value_name="ch4_mt")
        .assign(ghgi_ch4_kt=lambda df: df["ch4_mt"] / 1000)
        .drop(columns=["ch4_mt"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        .query("year.between(@min_year, @max_year)")
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
    )
    return emi_df


# %% STEP 2. Initialize Parameters
source_name = "petroleum systems"

proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_data = pd.read_excel(proxy_file_path, sheet_name="testing").query(
    f"gch4i_name == '{source_name}'"
)

emi_parameters_dict = {}
for emi_name, data in proxy_data.groupby("emi"):
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_name / x for x in data.file_name],
        "source_list": [x.strip().casefold() for x in data.ghgi_group.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_ww_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_petro_systems_inv_data(input_path, ghgi_group, parameters)
            emi_df_list.append(individual_emi_df)

        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        emission_group_df.to_csv(output_path)

# %% TESTING

# testing = get_petro_systems_inv_data(
#     in_path = emi_parameters_dict["oilwellprod_emi"]["input_paths"][1],
#     src = emi_parameters_dict["oilwellprod_emi"]["source_list"][1],
#     params = emi_parameters_dict["pet_hf_comp_emi"]["parameters"]
# )

# test = read_excel_params(proxy_file_path, subsector=source_name, emission="pneumatic devices", sheet='testing')

# ["chemical injection pumps", "pneumatic devices"]

# df = (pd.read_excel(proxy_file_path, sheet_name="testing")
#         .assign(
#             ghgi_group=lambda x: x['ghgi_group'].str.strip().str.casefold()
#             ))

# df = df.loc[df['gch4i_name'] == "petroleum systems"]

# df = df.loc[df['ghgi_group'] == "pneumatic devices", 'add_params']

# # Error is occuring because ghgi_group is not unique. Some share the same name.

# read_excel_params

# %%
