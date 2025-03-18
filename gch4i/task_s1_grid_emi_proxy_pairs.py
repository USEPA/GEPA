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
# %load_ext autoreload
# %autoreload 2
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
    QC_emi_raster_sums,
    QC_proxy_allocation,
    allocate_emissions_to_proxy,
    check_state_year_match,
    fill_missing_year_months,
    grid_allocated_emissions,
    make_emi_grid,
    normalize,
    scale_emi_to_month,
)

# import matplotlib.pyplot as plt
# import numpy as np
# from pytask import Product, mark, task


# # from typing import Annotated
# def fix_completed_status(status_db_path):
#     # quick fix for inconsistent labeling of completed status in the database
#     conn = sqlite3.connect(status_db_path)
#     cursor = conn.cursor()
#     cursor.execute(
#         """
#         UPDATE gridding_status
#         SET status = 'complete'
#         WHERE status = 'completed'
#         """
#     )
#     conn.commit()
#     conn.close()

# def remove_status(gch4i_name, emi_id, proxy_id):
#     cursor.execute(
#         """
#         DELETE FROM gridding_status
#         WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?
#         """,
#         (gch4i_name, emi_id, proxy_id),
#     )
#     conn.commit()


# Function to update the status of a row
def update_status(gch4i_name, emi_id, proxy_id, status):
    cursor.execute(
        """
    INSERT INTO gridding_status (gch4i_name, emi_id, proxy_id, status)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(gch4i_name, emi_id, proxy_id) DO UPDATE SET status=excluded.status
    """,
        (gch4i_name, emi_id, proxy_id, status),
    )
    conn.commit()


def get_status(gch4i_name, emi_id, proxy_id):
    cursor.execute(
        """
        SELECT status FROM gridding_status
        WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?
        """,
        (gch4i_name, emi_id, proxy_id),
    )
    result = cursor.fetchone()
    return result[0] if result else None


# from pytask import Product, task
gepa_profile = GEPA_spatial_profile()
out_profile = gepa_profile.profile

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("future.no_silent_downcasting", True)


def get_status_table(status_db_path, working_dir):
    # get a numan readable version of the status database
    conn = sqlite3.connect(status_db_path)
    status_df = pd.read_sql_query("SELECT * FROM gridding_status", conn)
    conn.close()
    status_df.to_csv(
        working_dir / f"gridding_status_{formatted_today}.csv", index=False
    )
    return status_df


# %%


# the mapping file that collections information needed for gridding on all emi/proxy
# pairs. This is the data driven approach to gridding that will be used in the
# production version of the gridding script.
mapping_file_path: Path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
v2_data_path: Path = V3_DATA_PATH.parents[1] / "v2_v3_comparison_crosswalk.csv"
# the path to the state and county shapefiles.
state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
county_geo_path: Path = global_data_dir_path / "tl_2020_us_county.zip"

# create the log file for today that writes out the status of all the gridding
# operations.
logger = logging.getLogger(__name__)
now = datetime.now()
formatted_today = now.strftime("%Y-%m-%d")
formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
working_dir = V3_DATA_PATH.parents[0] / "gridding_log_and_qc"
alloc_qc_dir = working_dir / "allocation_qc"
emi_grid_qc_dir = working_dir / "emi_grid_qc"
time_geo_qc_dir = working_dir / "time_geo_qc"
gridded_output_dir = working_dir / "gridded_output"

log_file_path = working_dir / f"gridding_log_{formatted_today}.log"
status_db_path = working_dir / "gridding_status.db"

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
state_gdf = (
    gpd.read_file(state_geo_path)
    .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
    .rename(columns=str.lower)
    .rename(columns={"stusps": "state_code", "name": "state_name"})
    .astype({"statefp": int})
    # get only lower 48 + DC
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .rename(columns={"statefp": "fips"})
    .to_crs(4326)
)

county_gdf = (
    gpd.read_file(county_geo_path)
    .loc[:, ["NAME", "STATEFP", "COUNTYFP", "geometry"]]
    .rename(columns=str.lower)
    .rename(columns={"name": "county_name"})
    .astype({"statefp": int, "countyfp": int})
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .assign(
        fips=lambda df: (
            df["statefp"].astype(str) + df["countyfp"].astype(str).str.zfill(3)
        ).astype(int)
    )
    .to_crs(4326)
)
# %%

geo_filter = state_gdf.state_code.unique().tolist() + ["OF"]
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
# if SKIP is set to True, the code will skip over any rows that have already been
# looked at based on the list of status values in the SKIP_THESE list.
# if SKIP is set to False, it will still check if the monthly or annual files exist and
# skip it. Otherwise it will try to run it again.
SKIP = True
SKIP_THESE = [
    "complete",
    # "failed state/year QC",
    # "error reading proxy",
    # "emi missing years",
    # "proxy file does not exist",
    # "proxy has emtpy geometries",
    # "failed raster QC",
    # "failed allocation QC",
    # "proxy year has NAs",
]

status_df = get_status_table(status_db_path, working_dir)
unique_pairs_df = mapping_df.drop_duplicates(
    subset=["gch4i_name", "emi_id", "proxy_id"]
)
unique_pairs_df = unique_pairs_df.merge(
    status_df, on=["gch4i_name", "emi_id", "proxy_id"]
)

if SKIP:
    unique_pairs_df = unique_pairs_df[~unique_pairs_df["status"].isin(SKIP_THESE)]
unique_pairs_df
# %%
conn = sqlite3.connect(status_db_path)
cursor = conn.cursor()
for row in tqdm(unique_pairs_df.itertuples(index=False), total=len(unique_pairs_df)):
    base_name = f"{row.gch4i_name}-{row.emi_id}-{row.proxy_id}"

    try:
        logging.info("=" * 83)
        logging.info(f"Gridding {base_name}.")

        # get the status of the row from the database. There will only be a status if the
        # file has not yet been done.

        annual_output_path = (
            gridded_output_dir / f"{row.gch4i_name}-{row.emi_id}-{row.proxy_id}.tif"
        )
        if row.emi_time_step == "monthly" or row.proxy_time_step == "monthly":
            monthly_output_path = (
                gridded_output_dir
                / f"{row.gch4i_name}-{row.emi_id}-{row.proxy_id}_monthly.tif"
            )
            # if annual_output_path.exists() and monthly_output_path.exists():
            #     logging.info(f"{base_name} already gridded.\n")
            #     update_status(row.gch4i_name, row.emi_id, row.proxy_id, "complete")
            #     continue
        # elif annual_output_path.exists():
        #     logging.info(f"{base_name} already gridded.\n")
        #     update_status(row.gch4i_name, row.emi_id, row.proxy_id, "complete")
        #     continue

        if not row.proxy_has_file:
            logging.critical(f"{row.proxy_id} proxy file does not exist.\n")
            update_status(
                row.gch4i_name, row.emi_id, row.proxy_id, "proxy file does not exist"
            )
            continue

        emi_input_path = list(emi_data_dir_path.glob(f"{row.emi_id}.csv"))[0]
        proxy_input_path = list(proxy_data_dir_path.glob(f"{row.proxy_id}.*"))[0]
        rel_emi_col = row.proxy_rel_emi_col
        # STEP 2: Read In EPA State/year GHGI Emissions
        # read the emi file and filter out the states that are not in the lower 48 + DC
        # and remove any records with 0 emissions.

        # there are 3 types of emi files:
        # 1. state+county/year+month/emissions
        # 2. state/year/emissions
        # 3. national/year/emissions

        logging.info(
            f"{row.emi_id} is at {row.emi_geo_level}/{row.emi_time_step} level."
        )
        logging.info(
            f"{row.proxy_id} is at {row.proxy_geo_level}/{row.proxy_time_step} level."
        )

        match row.emi_time_step:
            case "monthly":
                emi_time_cols = ["year", "month"]
            case "annual":
                emi_time_cols = ["year"]
            case _:
                logging.critical(f"emi_time_step {row.emi_time_step} not recognized")

        match row.emi_geo_level:
            case "national":
                emi_cols = emi_time_cols + ["ghgi_ch4_kt"]
                emi_df = pd.read_csv(emi_input_path, usecols=emi_cols)
            case "state":
                emi_cols = emi_time_cols + ["state_code", "ghgi_ch4_kt"]
                emi_df = pd.read_csv(emi_input_path, usecols=emi_cols).query(
                    "(state_code.isin(@geo_filter)) & (ghgi_ch4_kt > 0)"
                )
            case "county":
                emi_cols = emi_time_cols + ["state_code", "fips", "ghgi_ch4_kt"]
                emi_df = pd.read_csv(
                    emi_input_path,
                    usecols=emi_cols,
                ).query("(state_code.isin(@state_gdf.state_code)) & (ghgi_ch4_kt > 0)")
            case _:
                logging.critical(f"emi_geo_level {row.emi_geo_level} not recognized")

        if emi_df.year.unique().shape[0] != len(years):
            missing_years = list(set(years) - set(emi_df.year.unique()))
            if missing_years:
                logging.warning(f"{row.emi_id} is missing years: {missing_years}")
                logging.warning(f"these years will be filled with 0s.")
                # update_status(row.gch4i_name, row.emi_id, row.proxy_id, "emi missing years")
                # continue
        else:
            missing_years = False

        # if the emi file has a month column, we convert it a year-month column.
        # YYYY-MM
        if row.emi_time_step == "monthly":
            emi_df = emi_df.assign(
                month=lambda df: pd.to_datetime(df["month"], format="%B").dt.month,
                year_month=lambda df: pd.to_datetime(
                    df[["year", "month"]].assign(DAY=1)
                ).dt.strftime("%Y-%m"),
            )

        if row.emi_time_step == "monthly" or row.proxy_time_step == "monthly":
            time_col = "year_month"
        elif row.emi_time_step == "annual" and row.proxy_time_step == "annual":
            time_col = "year"

        if row.emi_geo_level == "national":
            geo_col = None
        elif row.emi_geo_level == "state":
            geo_col = "state_code"
        elif row.emi_geo_level == "county":
            geo_col = "fips"

        match_cols = [time_col, geo_col]
        match_cols = [x for x in match_cols if x is not None]

        # STEP 3: GET AND FORMAT PROXY DATA
        # we apply a different set of functions depending on the proxy type.
        # if vector type

        # STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES
        # embedded in this function is the check that the proxy has the same
        # year as the emissions and same set of states. It will log when that is
        # not the case since it is a critical error that needs to be corrected.
        if row.file_type == "parquet":

            # filter 1: national yearly emis

            # read the file, catch any errors and log them.
            try:
                proxy_gdf = gpd.read_parquet(proxy_input_path).reset_index()
            except Exception as e:
                update_status(
                    row.gch4i_name, row.emi_id, row.proxy_id, "error reading proxy"
                )
                logging.critical(f"Error reading {row.proxy_id}: {e}\n")
                continue

            if proxy_gdf.is_empty.any():
                update_status(
                    row.gch4i_name,
                    row.emi_id,
                    row.proxy_id,
                    "proxy has emtpy geometries",
                )
                logging.critical(f"{row.proxy_id} has empty geometries.\n")
                continue

            if not proxy_gdf.is_valid.all():
                update_status(
                    row.gch4i_name,
                    row.emi_id,
                    row.proxy_id,
                    "proxy has invalid geometries",
                )
                logging.critical(f"{row.proxy_id} has invalid geometries.\n")
                continue

            # minor formatting issue that some year_months in proxy data were written
            # with underscores instead of dashes.
            if row.proxy_has_year_month_col:
                proxy_gdf["year_month"] = pd.to_datetime(
                    proxy_gdf.year_month.str.replace("_", "-")
                ).dt.strftime("%Y-%m")

            # if the proxy doesn't have a year column, we explode the data out to
            # repeat the same data for every year.
            if not row.proxy_has_year_col:
                logging.info(f"{row.proxy_id} adding a year column.")
                # duplicate the data for all years in years_list
                proxy_gdf = proxy_gdf.assign(
                    year=lambda df: [years for _ in range(df.shape[0])]
                ).explode("year")
            else:
                try:
                    proxy_gdf = proxy_gdf.astype({"year": int})
                except:
                    logging.critical(f"{row.proxy_id} year column has NAs.\n")
                    update_status(
                        row.gch4i_name, row.emi_id, row.proxy_id, "proxy year has NAs"
                    )
                    continue

            # if the proxy data are monthly, but don't have the year_month column,
            # create it.
            if (row.proxy_time_step == "monthly") & (not row.proxy_has_year_month_col):
                logging.info(f"{row.proxy_id} adding a year_month column.")
                # add a year_month column to the proxy data
                try:
                    proxy_gdf = proxy_gdf.assign(
                        year_month=lambda df: pd.to_datetime(
                            df[["year", "month"]].assign(DAY=1)
                        ).dt.strftime("%Y-%m"),
                    )
                except ValueError:
                    proxy_gdf = proxy_gdf.assign(
                        month=lambda df: pd.to_datetime(
                            df["month"], format="%b"
                        ).dt.month,
                        year_month=lambda df: pd.to_datetime(
                            df[["year", "month"]].assign(DAY=1)
                        ).dt.strftime("%Y-%m"),
                    )

            # if the proxy data are monthly, but don't have the month column,
            # create it.
            if (row.proxy_time_step == "monthly") & (not row.proxy_has_month_col):
                logging.info(f"{row.proxy_id} adding a month column.")
                # add a month column to the proxy data
                proxy_gdf = proxy_gdf.assign(
                    month=lambda df: pd.to_datetime(df["year_month"]).dt.month
                )

            # if the proxy doesn't have relative emissions, we normalize by the
            # timestep of gridding (year, or year_month)
            if not row.proxy_has_rel_emi_col:
                logging.info(f"{row.proxy_id} adding a relative emissions column.")
                proxy_gdf["rel_emi"] = (
                    proxy_gdf.assign(emis_kt=1)
                    .groupby(match_cols)["emis_kt"]
                    .transform(normalize)
                )
                rel_emi_col = "rel_emi"

            # if the proxy file has monthly data, but the emi file does not, we expand
            # the emi file to match the year_month column of the proxy data.
            # We also have to calculate the monthly scaling factor from the relative
            # emissions of the proxy, versus equally allocating emissions temporally
            # across the year by just dividing by 12.
            if (row.proxy_time_step == "monthly") & (row.emi_time_step == "annual"):
                # break
                emi_df = scale_emi_to_month(proxy_gdf, emi_df, geo_col, time_col)

            # check that the proxy and emi files have matching state years
            # NOTE: this is going to arise when the proxy data are lacking adequate
            # spatial or temporal coverage. Failure here will require finding new data
            # and/or filling in missing state/years.
            time_geo_qc_pass, time_geo_qc_df = check_state_year_match(
                emi_df, proxy_gdf, row, geo_col, time_col
            )

            time_geo_qc_df.to_csv(time_geo_qc_dir / f"{base_name}_qc_state_year.csv")
            # if the proxy data do not match the emi, fail, continue out of this pair
            if time_geo_qc_pass:
                logging.info(f"QC PASSED: {base_name} passed state/year QC.")
            else:
                logging.critical(f"{base_name} failed state/year QC.\n")
                update_status(
                    row.gch4i_name, row.emi_id, row.proxy_id, "failed state/year QC"
                )
                continue

            # if we have passed the check that the proxy matches the emi by geographic
            # and temporal levels, we are ready to allocate the inventory emissions
            # to the proxy.
            allocation_gdf = allocate_emissions_to_proxy(
                proxy_gdf,
                emi_df,
                match_cols=match_cols,
                proportional_col_name=rel_emi_col,
            )

            # STEP X: QC ALLOCATION
            # check that the allocated emissions sum to the original emissions
            allocation_qc_pass, allocation_qc_df = QC_proxy_allocation(
                allocation_gdf,
                emi_df,
                row,
                geo_col,
                time_col,
                plot=False,
                plot_path=alloc_qc_dir / f"{base_name}_emi_compare.png",
            )
            allocation_qc_df.to_csv(alloc_qc_dir / f"{base_name}_allocation_qc.csv")
            # if the allocation failed, break the loop.
            # NOTE: if the process fails here, this is likely an issue with the
            # calculation of the proxy relative values (i.e. they do not all sum to 1
            # by the geo/time level)
            if allocation_qc_pass:
                logging.info(f"QC PASSED: {base_name} passed allocation QC.")
            else:
                logging.critical(f"{base_name} failed allocation QC.\n")
                update_status(
                    row.gch4i_name, row.emi_id, row.proxy_id, "failed allocation QC"
                )
                continue

            if allocation_gdf.empty:
                logging.warning(
                    f"{row.proxy_id} allocation is empty. likely no"
                    f"emissions data for {row.emi_id}"
                )
                proxy_times = proxy_gdf[time_col].unique()
                empty_shape = (
                    len(proxy_times),
                    gepa_profile.height,
                    gepa_profile.width,
                )
                empty_data = np.zeros(empty_shape)
                if time_col == "year_month":
                    proxy_ds = xr.DataArray(
                        empty_data,
                        dims=[time_col, "y", "x"],
                        coords={
                            time_col: proxy_times,
                            "y": gepa_profile.y,
                            "x": gepa_profile.x,
                        },
                        name="results",
                    ).to_dataset(name="results")
                    proxy_ds = proxy_ds.assign_coords(
                        year=(
                            "year_month",
                            pd.to_datetime(proxy_ds.year_month.values).year,
                        ),
                        month=(
                            "year_month",
                            pd.to_datetime(proxy_ds.year_month.values).month,
                        ),
                    )
                elif time_col == "year":
                    proxy_ds = xr.DataArray(
                        empty_data,
                        dims=[time_col, "y", "x"],
                        coords={
                            time_col: proxy_times,
                            "y": gepa_profile.y,
                            "x": gepa_profile.x,
                        },
                        name="results",
                    ).to_dataset(name="results")

            else:
                try:
                    # STEP X: GRID EMISSIONS
                    # turn the vector proxy into a grid
                    proxy_ds = grid_allocated_emissions(
                        allocation_gdf, timestep=time_col
                    )
                except Exception as e:
                    logging.critical(
                        f"{row.emi_id}, {row.proxy_id} gridding failed {e}\n"
                    )
                    update_status(
                        row.gch4i_name, row.emi_id, row.proxy_id, "gridding failed"
                    )
                    continue

        elif row.file_type == "netcdf":
            logging.info(f"{row.proxy_id} is a raster.")

            # read the proxy file
            proxy_ds = xr.open_dataset(proxy_input_path)  # .rename({"geoid": geo_col})
            # Drop any coordinates in proxy_ds that are not y, x, or the time_col
            proxy_ds = proxy_ds.drop_vars(
                [
                    coord
                    for coord in proxy_ds.coords
                    if coord not in [time_col, "y", "x"]
                ]
            )
            proxy_ds = proxy_ds.assign_coords(
                x=("x", gepa_profile.x), y=("y", gepa_profile.y)
            )
            # Reorder the dimensions of proxy_ds to be time, y, x
            # proxy_ds = proxy_ds.transpose(time_col, "y", "x").sortby([time_col, "y", "x"])

            if "fips" not in emi_df.columns:
                emi_df = emi_df.merge(
                    state_gdf[["state_code", "fips"]], on="state_code", how="left"
                )
            # if the emi is month and the proxy is annual, we expand the dimensions of
            # the proxy, repeating the year values for every month in the year
            # we stack the year/month dimensions into a single year_month so that
            # it aligns with the emissions data as a time x X x Y array.
            if row.emi_time_step == "monthly" and row.proxy_time_step == "annual":
                proxy_ds = (
                    proxy_ds.expand_dims(dim={"month": np.arange(1, 13)}, axis=0)
                    .stack({"year_month": ["year", "month"]})
                    .sortby(["year_month", "y", "x"])
                )
            # proxy_fips = np.unique(proxy_ds.fips.values)
            # emi_fips = np.unique(emi_df.fips.values)
            # missing_fips = set(emi_fips) - set(proxy_fips)
            # if len(missing_fips) != 0:
            #     logging.critical(f"{row.proxy_id} missing fips: {missing_fips}")

            match row.emi_geo_level:
                case "state":
                    admin_gdf = state_gdf
                case "county":
                    admin_gdf = county_gdf
            # NOTE: this is slow.
            emi_xr = make_emi_grid(emi_df, admin_gdf, time_col)

            # if the emi is missing a year by now, we need to fill in the missing years
            # we first remove the year from the proxy, then we add the missing years of
            # data as zero arrays at the end.

            # here we are going to remove missing years from the proxy dataset
            # so we can stack it against the emissions dataset
            if missing_years:
                proxy_ds = proxy_ds.drop_sel(year=list(missing_years))

            # assign the emissions array to the proxy dataset
            # proxy_ds["emissions"] = ([time_col, "y", "x"], emi_xr)
            proxy_ds["emissions"] = (emi_xr.dims, emi_xr.data)
            # # look to make sure the emissions loaded in the correct orientation
            # proxy_ds["emissions"].sel(year_month=(2020, 1)).where(lambda x: x > 0).plot(
            #     cmap="hot"
            # )
            # proxy_ds["emissions"].sel(year=2020).where(lambda x: x > 0).plot(cmap="hot")
            # plt.show()
            # calculate the emissions for the proxy array by multiplying the proxy by
            # the gridded emissions data
            proxy_ds["results"] = proxy_ds[rel_emi_col] * proxy_ds["emissions"]

        # The expectation at this step is that regardless of a parquet or gridded input
        # we have a proxy_ds that is a DataArray with dimensions of time, y, x
        # and a results variable that is the product of the proxy and the emissions.

        # in some cases, especially when we have a monthly proxy and annaul emissions,
        # we end up with months that do not exist. It is also possible for an emissions
        # source to be all zeros for a year, and also missing. So this function will
        # fill in the missing years and months with zeros.
        proxy_ds = fill_missing_year_months(proxy_ds, time_col)
        # if the time step is monthly, we need to also get yearly emissions data. We QC
        # the monthly data.
        if time_col == "year_month":
            yearly_results = proxy_ds["results"].groupby("year").sum()
            raster_qc_pass, raster_qc_df = QC_emi_raster_sums(
                proxy_ds["results"], emi_df, time_col
            )
            raster_qc_df.to_csv(
                emi_grid_qc_dir / f"{base_name}_emi_grid_qc_monthly.csv"
            )
            if not raster_qc_pass:
                logging.critical(f"{base_name} failed monthly raster QC.\n")
                update_status(
                    row.gch4i_name, row.emi_id, row.proxy_id, "failed monthly raster QC"
                )
                continue

            # if we have passed the QC step, we are ready to write out the monthly
            # gridded emissions data to a raster file.
            out_profile.update(count=len(years) * 12)
            out_arr = np.flip(
                proxy_ds["results"].transpose("year_month", "y", "x").values, axis=1
            )
            with rasterio.open(monthly_output_path, "w", **out_profile) as dst:
                dst.write(out_arr)
            logging.info(f"{base_name} monthly gridding complete.\n")
            update_status(row.gch4i_name, row.emi_id, row.proxy_id, "complete")
        else:
            # if the time step is annual, we need to get the annual emissions data. Here
            # this just gives it a new name, but it is the same as the results variable.
            yearly_results = proxy_ds["results"]
        # now we QC the annual data. This is the final QC step.
        raster_qc_pass, raster_qc_df = QC_emi_raster_sums(
            yearly_results, emi_df, "year"
        )
        raster_qc_df.to_csv(emi_grid_qc_dir / f"{base_name}_emi_grid_qc.csv")
        if raster_qc_pass:
            # if we have passed all the QC steps, we are ready to write out the annual
            # gridded emissions data to a raster file.
            out_profile.update(count=len(years))
            out_arr = np.flip(yearly_results.transpose("year", "y", "x").values, axis=1)
            with rasterio.open(annual_output_path, "w", **out_profile) as dst:
                dst.write(out_arr)
            logging.info(f"{base_name} annual gridding complete.\n")
            update_status(row.gch4i_name, row.emi_id, row.proxy_id, "complete")
        else:
            logging.critical(f"{base_name} failed annual raster QC.\n")
            update_status(
                row.gch4i_name, row.emi_id, row.proxy_id, "failed annual raster QC"
            )
            continue

    except Exception as e:
        logging.critical(f"{base_name} failed {e}\n")
        update_status(row.gch4i_name, row.emi_id, row.proxy_id, "failed")
        continue
conn.close()


# %%
