from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, mark, task

from gch4i.config import emi_data_dir_path, ghgi_data_dir_path, max_year, min_year
from gch4i.utils import tg_to_kt
import numpy as np

# %%
emi_list = [
    "maize",
    "rice",
    "wheat",
    "barley",
    "oats",
    "other_small_grains",
    "sorghum",
    "cotton",
    "grass_hay",
    "legume_hay",
    "peas",
    "sunflower",
    "tobacco",
    "vegetables",
    "chickpeas",
    "dry_beans",
    "lentils",
    "peanuts",
    "soybeans",
    "potatoes",
    "sugarbeets",
    "sugarcane",
]


def _get_kwargs(in_list):
    _id_to_kwargs = {}
    for item in in_list:
        _id_to_kwargs[item] = {
            "input_path": ghgi_data_dir_path / "field_burning/FBAR_90-22_State.xlsx",
            "emi_name": item,
            "output_path": emi_data_dir_path / f"{item}_emi.csv",
        }
    return _id_to_kwargs


_ID_TO_KWARGS = _get_kwargs(emi_list)
_ID_TO_KWARGS


# %%
for _id, _kwargs in _ID_TO_KWARGS.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_field_burning_emi(
        input_path: Path,
        emi_name: str,
        output_path: Annotated[Path, Product],
    ) -> None:
        """read in the ghgi_ch4_kt values for each state"""
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
            # drop columns we don't need
            # # get just methane emissions
            ## EEM: TODO - need to take the sum of liberated and recovered and used
            ##  in otherwords, the net methane emissions is the methane
            ## liberated minus the amount of methane recovered and used.
            .query("(ghg == 'CH4')")
            .drop(
                columns=[
                    "sector",
                    "category",
                    # "subcategory1",
                    # "subcategory2",
                    # "subcategory3",
                    "subcategory4",
                    "subcategory5",
                    "carbon pool",
                    "fuel1",
                    "fuel2",
                    "exclude",
                    "id",
                    "sensitive (y or n)",
                    "data type",
                    "subsector",
                    "crt code",
                    "units",
                    "ghg",
                    "gwp",
                ]
            )
            # set the index to state
            .assign(
                emi_group=lambda df: df["subcategory3"]
                .replace("", np.nan)
                .fillna(df["subcategory2"])
                .replace("", np.nan)
                .fillna(df["subcategory1"])
                # .replace("", np.nan)
                # .dropna()
                # .reset_index(drop=True)
                .str.replace(" ", "_")
                .str.lower()
                # .add("_emi")
            )
            .dropna(subset="emi_group")
            .drop(
                columns=[
                    "subcategory1",
                    "subcategory2",
                    "subcategory3",
                ]
            )
            .query("emi_group == @emi_name")
            .drop(columns="emi_group")
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
        )
        emi_df
        emi_df.to_csv(output_path, index=False)


# %%
task_field_burning_emi(**_kwargs)
# %%
