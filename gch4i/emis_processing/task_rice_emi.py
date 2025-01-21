"""
Name:                  3C Rice Cultivation Emissions
Date Last Modified:    2025-01-21
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               Clean and standardized rice cultivation emissions data
Input Files:           - gch4i_data_guide_v3.xlsx
                       - {V3_DATA_PATH}/ghgi/3C_rice_cultivation/Rice_90-22_State.xlsx
Output Files:          - {emi_data_dir_path}/rice_cult_emi.csv
"""

# %% Import Libraries
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


# %% Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name to filter the data guide
source_name = "3C_rice_cultivation"
# data guide directory
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# read and query for the source name (gch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

# initialize the emi_parameters_dict
emi_parameters_dict = {}
# loop through the proxy data and store the parameters in the emi_parameters_dict
for emi_name, data in proxy_data.groupby("emi_id"):
    emi_parameters_dict[emi_name] = {
        "input_path": ghgi_data_dir_path / source_name / data.file_name.iloc[0],
        "source_list": data.gch4i_source.to_list(),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
    }

# %% Create Pytask Function and Loop
for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_rice_fult_emi(
        input_path: Path,
        source_list: str,
        output_path: Annotated[Path, Product],
    ) -> None:
        """read in the ghgi_ch4_kt values for each state

        Args:
            input_path (Path): Path to the input data file.
            source_list (str): List of source names to extract from the data.
            output_path (Annotated[Path, Product]): Path to the output data file.

        Returns:
            pd.DataFrame: Processed emissions data with columns [state_code, year,
            ghgi_ch4_kt]
        """
        # Clean the source list
        source_list = [x.strip().casefold() for x in source_list]
        # Define the years of interest
        year_list = [str(x) for x in list(range(min_year, max_year + 1))]

        emi_df = (
            # read in the data
            pd.read_excel(
                input_path,
                sheet_name="InvDB",
                skiprows=15,
                # nrows=115,
                # usecols="A:BA",
            )
            # name column names lower
            .rename(columns=lambda x: str(x).lower())
            # format the names of emission source group strings
            .assign(
                ghgi_source=lambda df: df["category"]
                .astype(str)
                .str.strip()
                .str.casefold()
            )
            .dropna(subset="ghgi_source")
            # rename the location column to what we need
            .rename(columns={"georef": "state_code"})
            # get just CH4 emissions, get only the emissions of our ghgi group
            .query("(ghg == 'CH4') & (ghgi_source.isin(@source_list))")
            # get just the columns we need
            .filter(
                items=["state_code"] + year_list,
                axis=1,
            )
            .rename(columns={"georef": "state_code"})
            .set_index("state_code")
            # Convert NO/0 to NA, remove states with all 0 years
            .replace(0, pd.NA)
            .apply(pd.to_numeric, errors="coerce")
            .dropna(how="all")
            .fillna(0)
            # reset the index state back to a column
            .reset_index()
            # make the table long by state/year
            .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
            # convert units to kt
            .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
            # drop tg units column
            .drop(columns=["ch4_tg"])
            # make the columns types correcet
            .astype({"year": int, "ghgi_ch4_kt": float})
            .fillna({"ghgi_ch4_kt": 0})
            # get only the years we need
            .query("year.between(@min_year, @max_year)")
            # calculate a single value for each state/year
            # NOTE: applies when more than 1 source are being combined together.
            # otherwise has no effect.
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        # write out data to outoput path
        emi_df.to_csv(output_path, index=False)
