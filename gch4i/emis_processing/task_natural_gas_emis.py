# %%
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd

# from IPython.display import display
from pytask import Product, mark, task

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
)
from gch4i.utils import tg_to_kt

"""
Natural gas inventory emissions have 4 table forms. In the function, depending on the
read function required, and whether the data are state or national, the variables
read_function and groupers are assigned. Then the emissions groups can be run using
the code that reads each of the tables and groups them together and sums the values
by state [depending] and year.

TODO: there are gas inventory data that have not been delivered yet. We will need to
check if these new files fit into one of the existing 4 functions or if we need to write
a new function.

NOTE: the thing that works is that for each emissions group, all the input files are
read in the same exact way. This approach wouldn't work if input files assigned to a
single emissions group were read differently. Or else there would need to be another if
check against each input, which would be a pain.
"""


# %%
def read_dist_emi(in_path, sheet_name, src):
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    emi_df = (
        # read in the data
        pd.read_excel(
            in_path,
            sheet_name=sheet_name,
            skiprows=15,
            # nrows=115,
            # usecols="A:BA",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # rename the location column to what we need
        .rename(columns={"georef": "state_code"}).assign(
            ghgi_source=lambda df: df["subcategory1"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        # get just CH4 emissions, get only the emissions of our ghgi group
        .query(f"(ghg == 'CH4') & (ghgi_source == '{src}')")
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
        .melt(id_vars="state_code", var_name="year", value_name="ghgi_ch4_kt")
        # convert the units
        .assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * tg_to_kt)
        # drop the old units
        # .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ghgi_ch4_kt": float})
        # fill in missing values
        .fillna({"ghgi_ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
        # # calculate a single value for each state/year
        .reset_index()
    )
    return emi_df


def read_exp_term_data(in_path, sheet_name, src):
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]
    df = (
        pd.read_excel(in_path, sheet_name=sheet_name, header=5)
        .rename(columns=lambda x: str(x).lower())
        .assign(
            ghgi_source=lambda df: df["source"].astype(str).str.strip().str.casefold()
        )
        .query(f"ghgi_source == '{src}'")
        .filter(
            items=year_list,
            axis=1,
        )
        .melt(var_name="year", value_name="ch4_tg")
        # convert the units
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] / tg_to_kt)
        # drop the old units
        .drop(columns=["ch4_tg"])
    )
    return df


def read_allwell_data(in_path, sheet_name, src):
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    emi_df = (
        # read in the data
        pd.read_excel(
            in_path,
            sheet_name=sheet_name,
            # skiprows=15,
            nrows=57,
            # usecols="A:BA",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # format the names of the emission source group strings
        # .assign(
        #     ghgi_source=lambda df: df["subcategory1"]
        #     .astype(str)
        #     .str.strip()
        #     .str.casefold()
        # )
        # rename the location column to what we need
        .rename(columns={"state code": "state_code"})
        # get just CH4 emissions, get only the emissions of our ghgi group
        # .query(f"(ghg == 'CH4') & (ghgi_source == '{src}')")
        # get just the columns we need
        .filter(
            items=["state_code"] + year_list,
            axis=1,
        )
        # set the index to state
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .replace({0: np.nan})
        # # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        # convert the units
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] / tg_to_kt)
        # .rename(columns={"ch4_tg": "ghgi_ch4_kt"})
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
    emi_df.head()
    return emi_df


def read_postmeter_data(in_path, sheet_name, src):
    pass


# %%
source_name = "gas"
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

# TODO: need to come back when we have ALL the files for gas to remove the dropna
# and determine if new functions need to be written to read these new data or if they
# match one of the four existing read functions.
proxy_data = (
    pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping")
    .query(f"gch4i_name == '{source_name}'")
    .dropna(subset="file_name")
)

emi_parameters_dict = {}
for emi_name, data in proxy_data.groupby("emi"):

    input_paths = []
    for x in data.file_name:
        res = list(ghgi_data_dir_path.rglob(f"*{x}*"))
        if res:
            input_paths.append(res[0])
            emi_parameters_dict[emi_name] = {
                "input_paths": input_paths,
                "sheet_names": data["sheet name"].to_list(),
                "source_list": data.gch4i_source.to_list(),
                "output_path": emi_data_dir_path / f"{emi_name}.csv",
            }
        else:
            print(f"ERROR {x} doesn't exist")

emi_parameters_dict

# %%
# FOR TESTING
# _id = "dist_emi"
# # _id = "non_assoc_conv_emi"

# input_paths, sheet_names, source_list, output_path = emi_parameters_dict[_id].values()
# emi_name = output_path.stem
# source_list = [x.strip().casefold() for x in source_list]
# display(source_list)


# %%
for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_nat_gas_emis(
        input_paths: list[Path],
        sheet_names: list[str],
        source_list: list[str],
        output_path: Annotated[Path, Product],
    ) -> None:
        # get the emissions group name
        emi_name = output_path.stem
        # string format the source list to match formatting done in the read functions
        source_list = [x.strip().casefold() for x in source_list]

        # assign the read function based on the emissions group
        if emi_name in [
            "allwell_prod_emi",
            "tank_vent_emi",
            "non_assoc_exp_hf_comp_emi",
            "non_assoc_exp_conv_comp_emi",
            "basin_220_emi",
            "basin_395_emi",
            "basin_430_emi",
            "basin_other_emi",
            "federal_gom_offshore_emi",
            "gas_well_drilled_emi",
            "gb_stations_emi",
            "non_assoc_conv_emi",
            "non_assoc_exp_well_emi",
            "non_assoc_hf_emi",
            "not_mapped_emi",
            "prod_water_emi",
            "state_gom_offshore_emi",
            "well_blowout_emi",
        ]:
            read_function = read_allwell_data
            groupers = ["state_code", "year"]

        elif emi_name in ["dist_emi", "processing_emi"]:
            read_function = read_dist_emi
            groupers = ["state_code", "year"]

        elif emi_name in [
            "export_terminals_emi",
            "farm_pipelines_emi",
            "generators_emi",
            "import_terminals_emi",
            "lng_storage_emi",
            "storage_comp_station_emi",
            "storage_wells_emi",
            "trans_comp_station_emi",
            "trans_pipelines_emi",
        ]:
            read_function = read_exp_term_data
            groupers = ["year"]

        elif emi_name in [
            "postmeter_vehicles_emi",
            "postmeter_resi_emi",
            "postmeter_ind_egu_emi",
            "postmeter_comm_emi",
        ]:
            read_function = None
            groupers = ["state_code", "year"]
        else:
            return ValueError(f"{emi_name} not ready.")

        # for all inventory inputs to the emissions group, read the file and add to list
        emi_df_list = []
        for input_path, sheet_name, ghgi_group in zip(
            input_paths, sheet_names, source_list
        ):
            individual_emi_df = read_function(input_path, sheet_name, ghgi_group)

            emi_df_list.append(individual_emi_df)

        # concat all the input tables together and sum the emissions by state/year or
        # just year determined by groupers.
        emission_group_df = (
            pd.concat(emi_df_list).groupby(groupers)["ghgi_ch4_kt"].sum().reset_index()
        )
        # display (for testing)
        emission_group_df.head()
        # save the emi file.
        emission_group_df.to_csv(output_path)


# %%
"""
issues:

emi:                        issue:
postmeter_vehicles_emi      copy code from John
postmeter_resi_emi          copy code from John
postmeter_ind_egu_emi       copy code from John
postmeter_comm_emi          copy code from John

dist_emi                    working on code, but failing QC.
processing_emi              haven't done yet, new code
"""
# %%
