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
import sqlite3
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import osgeo  # noqa
import pandas as pd
import rasterio
import xarray as xr
from IPython.display import display
from pyarrow import parquet  # noqa
from rasterio.features import rasterize
from shapely import make_valid
from tqdm.auto import tqdm

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    global_data_dir_path,
    proxy_data_dir_path,
    years,
)
from gch4i.create_emi_proxy_mapping import get_gridding_mapping_df
from gch4i.utils import (
    GEPA_spatial_profile,
    # QC_emi_raster_sums,
    # QC_proxy_allocation,
    # allocate_emissions_to_proxy,
    # check_state_year_match,
    # fill_missing_year_months,
    # grid_allocated_emissions,
    # make_emi_grid,
    # scale_emi_to_month,
    # check_raster_proxy_time_geo,
    # write_tif_output,
)
from gch4i.gridding_utils import (
    EmiProxyGridder,
    get_status_table,
)

gepa_profile = GEPA_spatial_profile()
out_profile = gepa_profile.profile

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("future.no_silent_downcasting", True)
# %%


# the mapping file that collections information needed for gridding on all emi/proxy
# pairs. This is the data driven approach to gridding that will be used in the
# production version of the gridding script.
mapping_file_path: Path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
v2_data_path: Path = V3_DATA_PATH.parents[1] / "v2_v3_comparison_crosswalk.csv"
# the path to the state and county shapefiles.
# state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
# county_geo_path: Path = global_data_dir_path / "tl_2020_us_county.zip"

# create the log file for today that writes out the status of all the gridding
# operations.
logger = logging.getLogger(__name__)
now = datetime.now()
formatted_today = now.strftime("%Y-%m-%d")
formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

from gch4i.config import logging_dir

log_file_path = logging_dir / f"gridding_log_{formatted_today}.log"
status_db_path = logging_dir / "gridding_status.db"

logging.basicConfig(
    filename=log_file_path,
    encoding="utf-8",
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)


# %% STEP 1. Load GHGI-Proxy Mapping Files

# load the v2 file
v2_df = pd.read_csv(v2_data_path).rename(columns={"v3_gch4i_name": "gch4i_name"})
# %%

# collection all the information needed on the emi/proxy pairs for gridding
mapping_df = get_gridding_mapping_df(mapping_file_path)
# %%

mapping_df.set_index(["gch4i_name", "emi_id", "proxy_id"]).drop(
    columns="proxy_rel_emi_col"
).drop_duplicates(keep="last").reset_index(drop=True)

display(
    (
        mapping_df.groupby(
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

# we need this as a reference file to filter out the states that are not in the gridding
# region of lower 48 + DC.
# state_gdf = (
#     gpd.read_file(state_geo_path)
#     .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
#     .rename(columns=str.lower)
#     .rename(columns={"stusps": "state_code", "name": "state_name"})
#     .astype({"statefp": int})
#     # get only lower 48 + DC
#     .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
#     .rename(columns={"statefp": "fips"})
#     .to_crs(4326)
# )

# county_gdf = (
#     gpd.read_file(county_geo_path)
#     .loc[:, ["NAME", "STATEFP", "COUNTYFP", "geometry"]]
#     .rename(columns=str.lower)
#     .rename(columns={"name": "county_name"})
#     .astype({"statefp": int, "countyfp": int})
#     .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
#     .assign(
#         fips=lambda df: (
#             df["statefp"].astype(str) + df["countyfp"].astype(str).str.zfill(3)
#         ).astype(int)
#     )
#     .to_crs(4326)
# )
# geo_filter = state_gdf.state_code.unique().tolist() + ["OF"]
# %%

# %%
# Create a connection to the SQLite database

conn = sqlite3.connect(status_db_path)
cursor = conn.cursor()

# Create a table to store the status of each row
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS gridding_status (
    gch4i_name TEXT,
    emi_id TEXT,
    proxy_id TEXT,
    status TEXT,
    PRIMARY KEY (gch4i_name, emi_id, proxy_id)
)
"""
)
conn.commit()
# %%

status_df = get_status_table(status_db_path, logging_dir, formatted_today)
# %%
# if SKIP is set to True, the code will skip over any rows that have already been
# looked at based on the list of status values in the SKIP_THESE list.
# if SKIP is set to False, it will still check if the monthly or annual files exist and
# skip it. Otherwise it will try to run it again.
SKIP = False
SKIP_THESE = [
    "complete",
    "monthly failed, annual complete",
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
]
unique_pairs_df = mapping_df.drop_duplicates(
    subset=["gch4i_name", "emi_id", "proxy_id"]
)
unique_pairs_df = unique_pairs_df.merge(
    status_df, on=["gch4i_name", "emi_id", "proxy_id"]
)

if SKIP:
    unique_pairs_df = unique_pairs_df[~unique_pairs_df["status"].isin(SKIP_THESE)]
# unique_pairs_df = unique_pairs_df.query("gch4i_name == '1B1a_abandoned_coal'")
unique_pairs_df = unique_pairs_df.query("(gch4i_name == '1A_stationary_combustion')")
# unique_pairs_df = unique_pairs_df.query("(gch4i_name == '1A_stationary_combustion') & (emi_id == 'stat_comb_elec_oil_emi')")
unique_pairs_df
# %%

for row in tqdm(unique_pairs_df.itertuples(index=False), total=len(unique_pairs_df)):
    out_qc_dir = logging_dir / row.gch4i_name
    out_qc_dir.mkdir(exist_ok=True, parents=True)
    try:
        epg = EmiProxyGridder(row, status_db_path, out_qc_dir)
        epg.run_gridding()
        print(epg.base_name, epg.status)
    except Exception as e:
        print(epg.base_name, epg.status, e)

        
# %%
# alloc_gdf_proj = allocation_gdf.to_crs("ESRI:102003")
# intersect_mask = alloc_gdf_proj.intersects(cell_gdf.union_all())
# # %%
# ax = alloc_gdf_proj[intersect_mask].plot(color="xkcd:slate")
# alloc_gdf_proj[~intersect_mask].plot(color="xkcd:orange", markersize=15, ax=ax)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# # cell_gdf.plot(ax=ax, color="xkcd:lightblue", alpha=0.5)
# state_gdf.plot(ax=ax, color="xkcd:lightgreen", alpha=0.5)
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)

# # %%
# ax = allocation_gdf[intersect_mask].plot(color="xkcd:slate")
# allocation_gdf[~intersect_mask].plot(color="xkcd:orange", markersize=15, ax=ax)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# # cell_gdf.plot(ax=ax, color="xkcd:lightblue", alpha=0.5)
# state_gdf.plot(ax=ax, color="xkcd:lightgreen", alpha=0.5)
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# # %%
# proxy_intersect_mask = proxy_gdf.intersects(state_gdf.union_all())

# ax = proxy_gdf[proxy_intersect_mask].plot(color="xkcd:slate")
# proxy_gdf[~proxy_intersect_mask].plot(color="xkcd:orange", markersize=15, ax=ax)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# # cell_gdf.plot(ax=ax, color="xkcd:lightblue", alpha=0.5)
# state_gdf.plot(ax=ax, color="xkcd:lightgreen", alpha=0.5)
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# # %%
# import matplotlib.pyplot as plt

# state_union = state_gdf.union_all()

# # %%
# from gch4i.utils import get_cell_gdf

# cell_gdf = get_cell_gdf()
# cell_gdf_union = cell_gdf.dissolve().to_crs(4326)

# # %%

# data_dict = {}
# for row in tqdm(unique_pairs_df.itertuples(index=False), total=len(unique_pairs_df)):
#     if row.file_type == "parquet":
#         proxy_input_path = list(proxy_data_dir_path.glob(f"{row.proxy_id}.*"))[0]
#         proxy_gdf = gpd.read_parquet(proxy_input_path).reset_index()
#         print(f"{row.proxy_id} has {proxy_gdf.shape[0]} rows.")
#         data_dict[row.proxy_id] = proxy_gdf

# # %%
# for proxy_id, proxy_gdf in data_dict.items():
#     if "state_code" in proxy_gdf.columns:
#         state_mask = proxy_gdf.state_code.isin(state_gdf.state_code.unique())
#         proxy_gdf = proxy_gdf[state_mask]
#         proxy_out_of_state = proxy_gdf[~state_mask]
#         display(proxy_out_of_state)
#         print(f"{proxy_id} has {proxy_out_of_state.shape[0]} rows outside of states.")
#         proxy_intersect_mask = proxy_gdf.intersects(state_union)
#         bad_data_gdf = proxy_gdf[~proxy_intersect_mask]
#         good_data_gdf = proxy_gdf[proxy_intersect_mask]
#         outside_states = bad_data_gdf.state_code.value_counts()
#         print(outside_states)
#     else:
#         bad_data_gdf = proxy_gdf[~proxy_intersect_mask]
#         good_data_gdf = proxy_gdf[proxy_intersect_mask]

#     print(f"there are {bad_data_gdf.shape[0]} bad records")

#     if not proxy_intersect_mask.all():
#         ax = good_data_gdf.plot(color="xkcd:slate", markersize=1, zorder=10)
#         if "state_code" in bad_data_gdf.columns:
#             bad_data_gdf.plot(color="xkcd:scarlet", legend=False, markersize=15, ax=ax)
#         else:
#             bad_data_gdf.plot(color="xkcd:scarlet", markersize=15, ax=ax)
#         # xlim = ax.get_xlim()
#         # ylim = ax.get_ylim()
#         # cell_gdf.plot(ax=ax, color="xkcd:lightblue", alpha=0.5)
#         state_gdf.plot(ax=ax, color="xkcd:lightgreen", alpha=0.5)
#         cell_gdf_union.plot(ax=ax, color="xkcd:lightblue", alpha=0.5)
#         ax.set_title(f"{proxy_id} has {bad_data_gdf.shape[0]} bad records")
#         # ax.set_xlim(xlim)
#         # ax.set_ylim(ylim)
#         plt.show()
#         plt.close()
#     print()
# # %%
# tmp_pairs_df = mapping_df.drop_duplicates(
#     subset=["gch4i_name", "emi_id", "proxy_id"]
# ).merge(
#     status_df, on=["gch4i_name", "emi_id", "proxy_id"]
# ).query("gch4i_name == '1A_stationary_combustion'")
# tmp_pairs_df
# # %%
