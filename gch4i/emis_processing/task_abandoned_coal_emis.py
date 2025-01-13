"""
Name:                  Abandoned_Coal_Emissions
Date Last Modified:    2024-09-19
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               Clean and standardized abandoned coal methane emissions data
Input Files:           - gch4i_data_guide_v3.xlsx
                       - {V3_DATA_PATH}/ghgi/1B1a_abandoned_coal/AbandonedCoalMines1990-2022_FRv1.xlsx
Output Files:          - {emi_data_dir_path}/abd_coal_emi.csv
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
)
from gch4i.utils import tg_to_kt


def read_emi_data(in_path, src):
    """
    Read and process abandoned coal mines methane emissions data from given filepath.
    
    Args:
        in_path (Path): Path to input Excel file
        src (str): Source category string to filter data
        
    Returns:
        pd.DataFrame: Processed emissions data with columns [state_code, year, ghgi_ch4_kt]
    """
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    emi_df = (
        # read in the data
        pd.read_excel(
            in_path,
            sheet_name="InvDB",
            skiprows=15,
            # nrows=115,
            # usecols="A:BA",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # format the names of the emission source group strings
        .assign(
            ghgi_source=lambda df: df["subcategory1"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        # rename the location column to what we need
        .rename(columns={"georef": "state_code"})
        # get just CH4 emissions, get only the emissions of our ghgi group
        .query(
            f"(ghg == 'CH4') & (ghgi_source == '{src}') & (state_code != 'National')"
        )
        # get just the columns we need
        .filter(
            items=["state_code"] + year_list,
            axis=1,
        )
        # set the index to state
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        # convert the units
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        # drop the old units
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ghgi_ch4_kt": float})
        # fill in missing values
        .fillna({"ghgi_ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
        # calculate a single value for each state/year
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
    )
    return emi_df


# Retrieve input data filepath from data guide sheet
source_name = "abandoned_coal" #TODO: update (1B1a_abandoned_coal ?)
data_guide_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
proxy_data = pd.read_excel(data_guide_path, sheet_name="testing").query( #TODO: Update sheet name. (emi_proxy_mapping ?)
    f"gch4i_name == '{source_name}'"
)
emi_parameters_dict = {}
for emi_name, data in proxy_data.groupby("emi"):
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_name / x for x in data.file_name],
        "source_list": data.ghgi_group.to_list(),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
    }

# loop over input data files and process each using read_emi_data
for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_and_coal_emi(
        input_paths: list[Path],
        source_list: list[str],
        output_path: Annotated[Path, Product],
    ) -> None:

        source_list = [x.strip().casefold() for x in source_list]

        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = read_emi_data(input_path, ghgi_group)
            emi_df_list.append(individual_emi_df)

        emission_group_df = ( # Group data from different input files into single table
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        emission_group_df.to_csv(output_path)
