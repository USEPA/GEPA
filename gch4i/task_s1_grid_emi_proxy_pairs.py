"""
Name:                   test_gch4i_standard_gridding.py
Date Last Modified:     2024-06-07
Authors Name:           N. Kruskamp, H. Lohman (RTI International), Erin McDuffie
                        (EPA/OAP)
Purpose:                This is a template framework for a gridder that applies
                        to every gch4i source that uses a one-to-one emissions to proxy
                        allocation. The user were specify the list of gch4i gridding
                        sources and the gridder would be applied to each one of them,
                        pairing the emission and proxies and other argumentes needed
                        from the proxy mapping spreadsheet.
Input Files:            -
Output Files:           -
Notes:                  - Currently this should handle all the "standard" emi-proxy
                        pairs where the emi has a year, state_code and, ghgi_ch4_kt
                        column, and the proxy has a year, state_code, rel_emi, and
                        geometry colum. This should with a few modifications also handle
                        the proxies that do not have years (the same data are repeated
                        every year) and/or no relative emissions (where we assumed
                        uniform distribution of state level emissions to all proxy
                        rows).
                        TODO: cotinue to add more flexibility to handle other
                        gridding sources where the timesteps do not match between the
                        emi and proxy (e.g. emi has year+month, proxy has year only,
                        or vice versa).
                        TODO: the way I've written this, I wanted to update the code so
                        that the timestep columns were passed to the allocation function
                        and were used as grouper columns, instead of having year
                        hardcoded into the fucntion
                        TODO: write the emi to raster proxy function
                        TODO: update the raster QC function to do state+year QC instead
                        of just year.
                        TODO: probably in a follow-on script to this one, we need the
                        final netcdf formatter script that puts all these output files
                        into the yearly netcdf files.
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
# for testing/development
%load_ext autoreload
%autoreload 2
# %%

import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

from gch4i.config import V3_DATA_PATH, logging_dir
from gch4i.gridding_utils import EmiProxyGridder, GriddingMapper, get_status_table

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("future.no_silent_downcasting", True)
# create the log file for today that writes out the status of all the gridding
# operations.
logger = logging.getLogger(__name__)
now = datetime.now()
formatted_today = now.strftime("%Y-%m-%d")
formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
# %%


# the mapping file that collections information needed for gridding on all emi/proxy
# pairs. This is the data driven approach to gridding that will be used in the
# production version of the gridding script.
status_db_path = logging_dir / "gridding_status.db"
v2_data_path: Path = V3_DATA_PATH.parents[1] / "v2_v3_comparison_crosswalk.csv"
mapping_file_path: Path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
log_file_path = logging_dir / f"gridding_log_{formatted_today}.log"
# start the log file
logging.basicConfig(
    filename=log_file_path,
    encoding="utf-8",
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
# collection all the information needed on the emi/proxy pairs for gridding
# %%
g_mapper = GriddingMapper(mapping_file_path)
status_df = get_status_table(status_db_path, logging_dir, formatted_today, save=True)

g_mapper.mapping_df.set_index(["gch4i_name", "emi_id", "proxy_id"]).drop(
    columns="proxy_rel_emi_col"
).drop_duplicates(keep="last").reset_index(drop=True)

display(
    (
        g_mapper.mapping_df.groupby(
            [
                "file_type",
                "emi_time_step",
                "proxy_time_step",
                "emi_geo_level",
                "proxy_geo_level",
            ]
        )
        .size()
        .rename("pair_count")
        .reset_index()
    )
)
# %%
# if SKIP is set to True, the code will skip over any rows that have already been
# looked at based on the list of status values in the SKIP_THESE list.
# if SKIP is set to False, it will still check if the monthly or annual files exist and
# skip it. Otherwise it will try to run it again.
SKIP = False
SKIP_THESE = [
    # "complete",
    # "monthly failed, annual complete",
    # "failed state/year QC",
    # "error reading proxy",
    # "emi missing years",
    # "proxy file does not exist",
    # "proxy has emtpy geometries",
    # "failed raster QC",
    # "failed allocation QC",
    # "proxy year has NAs",
    # "failed annual raster QC",
    # "failed monthly raster QC",
    # "failed",
]
ready_for_gridding_df = g_mapper.mapping_df.drop_duplicates(
    subset=["gch4i_name", "emi_id", "proxy_id"]
)
ready_for_gridding_df = ready_for_gridding_df.merge(
    status_df, on=["gch4i_name", "emi_id", "proxy_id"]
)

if SKIP:
    ready_for_gridding_df = ready_for_gridding_df[
        ~ready_for_gridding_df["status"].isin(SKIP_THESE)
    ]
ready_for_gridding_df = ready_for_gridding_df.query(
    "(gch4i_name == '3B_manure_management')"
#     " | (gch4i_name == '5D2_industrial_wastewater')"
)
ready_for_gridding_df
# %%
# example for running all emi/proxy pairs
for row in tqdm(
    ready_for_gridding_df.itertuples(index=False), total=len(ready_for_gridding_df)
):
    out_qc_dir = logging_dir / row.gch4i_name
    try:
        epg = EmiProxyGridder(row, status_db_path, out_qc_dir)
        epg.run_gridding()
        print(epg.base_name, epg.status)
    except Exception as e:
        print(epg.base_name, epg.status, e)
# %%
# example for running a single emi/proxy pair
gch4i_name = "3A_enteric_fermentation"
emi_id = "enteric_fermentation_swine_emi"
proxy_id = "livestock_swine_proxy"
row = ready_for_gridding_df.query(
    f"gch4i_name == '{gch4i_name}' & emi_id == '{emi_id}' & proxy_id == '{proxy_id}'"
).iloc[0]

out_qc_dir = logging_dir / row.gch4i_name
epg = EmiProxyGridder(row, status_db_path, out_qc_dir)
epg.run_gridding()
# %%
import rasterio
# %%
with rasterio.open(epg.annual_output_path) as src:
    arr = src.read()
arr.shape
# %%
epg.proxy_ds["results"].transpose(epg.time_col, "y", "x")
# %%
