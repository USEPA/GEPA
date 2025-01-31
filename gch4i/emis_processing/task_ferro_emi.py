"""
Name:                  2C2 Ferroalloy Emissions
Date Last Modified:    2025-01-17
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               Clean and standardized abandoned ferroalloy emissions data
Input Files:           - gch4i_data_guide_v3.xlsx
                       - {V3_DATA_PATH}/ghgi/2C2_ferroalloy/State_Ferroalloys_1990-2022.
                        xlsx
Output Files:          - {emi_data_dir_path}/ferro_emi.csv
"""

# %%
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
)
from gch4i.utils import tg_to_kt


def read_emi_data(in_path, src):
    """
    Read and process ferroalloy emissions data from the given filepath.

    Args:
        in_path (Path): Path to input Excel file
        src (str): Source category string to filter data

    Returns:
        pd.DataFrame: Processed emissions data with columns [state_code, year,
        ghgi_ch4_kt]
    """
    # Define years of interest
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
            ghgi_source=lambda df: df["category"].astype(str).str.strip().str.casefold()
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


# %% Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name to filter the data guide
source_name = "2C2_ferroalloy"
# data guide Directory
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
        "input_paths": [list(ghgi_data_dir_path.rglob(x))[0] for x in data.file_name],
        "source_list": data.gch4i_source.to_list(),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
    }

# %% Create Pytask Function and Loop
for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_ferro_emi_data(
        input_paths: list[Path],
        source_list: list[str],
        output_path: Annotated[Path, Product],
    ) -> None:
        """read in the ghgi_ch4_kt values for each state"""

        source_list = [x.strip().casefold() for x in source_list]

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to ge the emissions data
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = read_emi_data(input_path, ghgi_group)
            emi_df_list.append(individual_emi_df)

        # Concatenate the emissions data and group by state and year
        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        # Save the emissions data to the output path
        emission_group_df.to_csv(output_path)
