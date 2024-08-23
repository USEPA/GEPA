"""
Name:                   task_petro_refining_emi.py
Date Last Modified:     2024-08-22
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of petroleum systems emissions
                        to State, Year, emissions format
Input Files:            1B2aiv_petroleum_refining
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

from gch4i.utils import tg_to_kt

# %% STEP 1. Create Emi Mapping Functions


def get_petro_refining_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    Subcategory Dictionary:
    Crude Oil Refining
    """

    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Sheet name
        skiprows=params["arguments"][1],  # number of rows to skip
        nrows=params["arguments"][2],  # number of rows
        index_col=None
        )

    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    emi_df = (
        emi_df.rename(columns=lambda x: str(x).lower())
        .query('ghg == "CH4" and subcategory1 == "Crude Oil Refining"')
        .rename(columns={"georef": "state_code"})
        .filter(items=["state_code"] + year_list, axis=1)
        .set_index("state_code")
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ghgi_ch4_kt=lambda x: x["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        .query("year.between(@min_year, @max_year)")
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
    )
    return emi_df


# %% STEP 2. Initialize Parameters
source_name = "1B2aiv_petroleum_refining"
source_path = "Petroleum and Natural Gas"

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
    def task_petro_refining_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_petro_refining_inv_data(input_path, ghgi_group, parameters)
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

# testing = get_petro_refining_inv_data(
#     in_path=emi_parameters_dict["refining_emi"]["input_paths"][0],
#     src=emi_parameters_dict["refining_emi"]["source_list"][0],
#     params=emi_parameters_dict["refining_emi"]["parameters"]
# )


# emi_df = pd.read_excel(
#     io = emi_parameters_dict["refining_emi"]["input_paths"][0],
#     sheet_name=emi_parameters_dict["refining_emi"]["parameters"]["arguments"][0],  # Sheet name
#     skiprows=emi_parameters_dict["refining_emi"]["parameters"]["arguments"][1],  # number of rows to skip
#     nrows=emi_parameters_dict["refining_emi"]["parameters"]["arguments"][2],  # number of rows
#     index_col=None
#     )

# emi_df = (
#     emi_df.rename(columns=lambda x: str(x).lower())
#     .query('ghg == "CH4" and subcategory1 == "Crude Oil Refining"')
# )




# test_path = "/Users/aburnette/Library/CloudStorage/OneDrive-SharedLibraries-EnvironmentalProtectionAgency(EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/ghgi/Petroleum and Natural Gas/Tanks_StateEstimates.xlsx"
# test = pd.read_excel(
#     io = test_path,
#     sheet_name = "Petro_State_CH4_MT",
#     nrows = 56,
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
