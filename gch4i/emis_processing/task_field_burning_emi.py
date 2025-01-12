from pathlib import Path
from typing import Annotated

import numpy as np
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

# %%
source_name = "3F4_fbar"

proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)
proxy_data

emi_parameters_dict = {}

for emi_name, data in proxy_data.groupby("emi_id"):
    emi_parameters_dict[emi_name] = {
        "input_path": ghgi_data_dir_path / source_name / data.file_name.iloc[0],
        "source_list": data.gch4i_source.to_list(),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
    }

emi_parameters_dict

input_path, source_list, output_path = emi_parameters_dict["wheat_emi"].values()
input_path.exists()


# %%
for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_field_burning_emi(
        input_path: Path,
        source_list: str,
        output_path: Annotated[Path, Product],
    ) -> None:

        source_list = [x.strip().casefold() for x in source_list]
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
            .assign(
                ghgi_source=lambda df: df["subcategory3"]
                .replace("", np.nan)
                .fillna(df["subcategory2"])
                .replace("", np.nan)
                .fillna(df["subcategory1"])
                .str.replace(" ", "_")
                .str.lower()
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
            # covert "NO" string to numeric (will become np.nan)
            .apply(pd.to_numeric, errors="coerce")
            # drop states that have all nan values
            .dropna(how="all")
            # reset the index state back to a column
            .reset_index()
            # make the table long by state/year
            .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
            .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
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
        emi_df
        emi_df.to_csv(output_path, index=False)


# %%
