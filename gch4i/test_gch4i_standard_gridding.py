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

from pyarrow import parquet  # noqa
import osgeo  # noqa
import geopandas as gpd
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm
import numpy as np

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    global_data_dir_path,
    proxy_data_dir_path,
    tmp_data_dir_path,
    years,
)
from rasterio.features import rasterize
from gch4i.create_emi_proxy_mapping import get_gridding_mapping_df
from gch4i.utils import (
    QC_emi_raster_sums,
    QC_flux_emis,
    QC_proxy_allocation,
    allocate_emis_to_array,
    allocate_emissions_to_proxy,
    calculate_flux,
    combine_gridded_emissions,
    grid_allocated_emissions,
    plot_annual_raster_data,
    plot_raster_data_difference,
    write_ncdf_output,
    write_tif_output,
    check_state_year_match,
    normalize,
    scale_emi_to_month,
    make_emi_grid
)

# import matplotlib.pyplot as plt
# import numpy as np
# from pytask import Product, mark, task


# from typing import Annotated


# from pytask import Product, task


gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
logger = logging.getLogger(__name__)
now = datetime.now()
formatted_today = now.strftime("%Y-%m-%d")
formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
# %%
log_file_path = (
    V3_DATA_PATH.parents[0] / "gridding_logs" / f"gridding_log_{formatted_today}.log"
)
print(formatted_datetime)
logging.basicConfig(
    filename=log_file_path,
    encoding="utf-8",
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)


# %% STEP 1. Load GHGI-Proxy Mapping Files

# this currently comes out of the prototype test_emi_proxy_mapping.py script
# I think this data driven approach to emi proxy mapping would be a good step into
# gridding automation.
mapping_df = get_gridding_mapping_df()
mapping_df

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


def create_gridding_params(in_df):

    # this formats the arguments to the gridding function
    gridding_params = {}
    for gch4i_name, data in in_df.groupby("gch4i_name"):

        GRIDDING_NAME = data.gch4i_name.iloc[0].lower()
        # v2_name = data.v2_name.iloc[0]
        IPCC_ID, SOURCE_NAME = gch4i_name.split("_", maxsplit=1)

        netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
        netcdf_description = (
            f"Gridded EPA Inventory - {GRIDDING_NAME} - "
            f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
        )

        gridding_params[gch4i_name] = dict(
            source_name=gch4i_name,
            group_data=data,
            nc_flux_output_path=tmp_data_dir_path / f"{GRIDDING_NAME}_ch4_emi_flux.nc",
            nc_kt_output_path=tmp_data_dir_path / f"{GRIDDING_NAME}_ch4_kt_per_year.nc",
            tif_flux_output_path=tmp_data_dir_path
            / f"{GRIDDING_NAME}_ch4_emi_flux.tif",
            tif_kt_output_path=tmp_data_dir_path
            / f"{GRIDDING_NAME}_ch4_kt_per_year.tif",
            netcdf_title=netcdf_title,
            netcdf_description=netcdf_description,
            # "annual_plot_output_path = "",
            # "diff_plot_output_path = "",
        )
    return gridding_params


gridding_params = create_gridding_params(mapping_df)

# %%

# we need this as a reference file to filter out the states that are not in the gridding
# region of lower 48 + DC.
state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
county_geo_path: Path = global_data_dir_path / "tl_2020_us_county.zip"

state_gdf = (
    gpd.read_file(state_geo_path)
    .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
    .rename(columns=str.lower)
    .rename(columns={"stusps": "state_code", "name": "state_name"})
    .astype({"statefp": int})
    # get only lower 48 + DC
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
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

# once this is ready for production, we would adjust this loop to take in the
# gridding_params and apply the gridding function to each one of them.
# for _id, kwargs in gridding_params.items():

#     @mark.persist
#     @task(id=_id, kwargs=kwargs)
#     def task_gridding(
#         source_name: str,
#         emi_input_paths: list[Path],
#         proxy_input_paths: list[Path],
#         proxy_type_list: list[str],
#         v2_name: str,
#         nc_flux_output_path: Annotated[Path, Product],
#         nc_kt_output_path: Annotated[Path, Product],
#     ) -> None:


# (
#     source_name,
#     emi_input_paths,
#     emi_timestep_list,
#     proxy_input_paths,
#     proxy_timtestep_list,
#     proxy_has_prop_list,
#     proxy_prop_col_list,
#     nc_flux_output_path,
#     nc_kt_output_path,
#     tif_flux_output_path,
#     tif_kt_output_path,
# ) = gridding_params['2B5_carbide'].values()

# this is the manual version of the above pytask approach for testing.
# this looks over each gridding group, and there is a inner loop that goes through
# the emi/proxy pairs.
logging.info(f"Started gridding at {formatted_datetime}")
N_SOURCES_READY = len(gridding_params)
for i, (_id, kwargs) in tqdm(
    enumerate(gridding_params.items(), 1), total=N_SOURCES_READY
):

    (
        source_name,
        data,
        nc_flux_output_path,
        nc_kt_output_path,
        tif_flux_output_path,
        tif_kt_output_path,
        netcdf_title,
        netcdf_description,
    ) = kwargs.values()

    logging.info(
        f"\n\n#{i}/{N_SOURCES_READY} ========================================"
        f"\nStarting {source_name}"
    )

    file_status = [
        x.exists()
        for x in [
            nc_flux_output_path,
            # nc_kt_output_path,
            tif_flux_output_path,
            tif_kt_output_path,
        ]
    ]

    # if all(file_status):
    #     logging.info(f"{source_name} already gridded.\n")
    #     continue

    # a dictionary to hold the gridding group gridding results for all emi/proxy pairs
    # %%
    source_group_dict = {}
    # for each emi/proxy pair in the gridding group
    for row in data.itertuples(index=False):
        # break
        logging.info(f"Processing {row.emi_id}, {row.proxy_id}")
        # add key to dictionary for this emi/proxy pair
        source_group_dict[row.emi_id] = {}
        emi_input_path = list(emi_data_dir_path.glob(f"{row.emi_id}.csv"))[0]
        proxy_input_path = list(proxy_data_dir_path.glob(f"{row.proxy_id}.*"))[0]
        # break

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
                    "(state_code.isin(@state_gdf.state_code)) & (ghgi_ch4_kt > 0)"
                )
            case "county":
                emi_cols = emi_time_cols + ["state_code", "fips", "ghgi_ch4_kt"]
                emi_df = pd.read_csv(
                    emi_input_path,
                    usecols=emi_cols,
                ).query("(state_code.isin(@state_gdf.state_code)) & (ghgi_ch4_kt > 0)")
            case _:
                logging.critical(f"emi_geo_level {row.emi_geo_level} not recognized")

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

        # match row.proxy_time_step:
        #     case "monthly":
        #         if row.proxy_has_year_col & row.proxy_has_month_col:
        #             proxy_timestep = ["year", "month"]
        #         elif (not row.proxy_has_year_col) & row.proxy_has_month_col:
        #             proxy_timestep = ["month"]
        #         elif row.proxy_has_year_month_col:
        #             proxy_timestep = ["year_month"]
        #         else:
        #             logging.critical(
        #                 f"{row.proxy_id} has neither year nor month columns."
        #             )
        #     case "annual":
        #         proxy_timestep = ["year"]
        #     case _:
        #         logging.critical(
        #             f"proxy_time_step {row.proxy_time_step} not recognized"
        #         )

        # STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES
        # embedded in this function is the check that the proxy has the same
        # year as the emissions and same set of states. It will log when that is
        # not the case since it is a critical error that needs to be corrected.
        if row.file_type == "parquet":

            # filter 1: national yearly emis

            # read the file, catch any errors and log them.
            try:
                proxy_gdf = gpd.read_parquet(proxy_input_path)
            except Exception as e:
                logging.critical(f"Error reading {row.proxy_id}: {e}")
                continue

            # if the proxy doesn't have a year column, we explode the data out to
            # repeat the same data for every year.
            if not row.proxy_has_year_col:
                logging.info(f"{row.proxy_id} adding a year column.")
                # duplicate the data for all years in years_list
                proxy_gdf = proxy_gdf.assign(
                    year=lambda df: [years for _ in range(df.shape[0])]
                ).explode("year")

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
                        month=lambda df: pd.to_datetime(df["month"], format="%b").dt.month,
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
                row.proxy_rel_emi_col = "rel_emi"

            # if the proxy file has monthly data, but the emi file does not, we expand
            # the emi file to match the year_month column of the proxy data.
            # We also have to calculate the monthly scaling factor from the relative
            # emissions of the proxy, versus equally allocating emissions temporally
            # across the year by just dividing by 12.
            if (row.proxy_time_step == "monthly") & (row.emi_time_step == "annual"):
                emi_df = scale_emi_to_month(proxy_gdf, emi_df, row)

            # check that the proxy and emi files have matching state years
            # NOTE: this is going to arise when the proxy data are lacking adequate
            # spatial or temporal coverage. Failure here will require finding new data
            # and/or filling in missing state/years.
            time_geo_qc_pass = check_state_year_match(
                emi_df, proxy_gdf, row, match_cols=match_cols
            )
            # if the proxy data do not match the emi, fail, break out of this pair
            if not time_geo_qc_pass:
                break

            # if we have passed the check that the proxy matches the emi by geographic
            # and temporal levels, we are ready to allocate the inventory emissions
            # to the proxy.
            allocation_gdf = allocate_emissions_to_proxy(
                proxy_gdf,
                emi_df,
                # proxy_has_year=row.proxy_has_year_col,
                match_cols=match_cols,
                # use_proportional=row.proxy_has_rel_emi_col,
                proportional_col_name=row.proxy_rel_emi_col,
            )

            # STEP X: QC ALLOCATION
            # check that the allocated emissions sum to the original emissions
            allocation_qc_pass = QC_proxy_allocation(
                allocation_gdf, emi_df, row, geo_col=geo_col, time_col=time_col
            )
            # if the allocation failed, break the loop.
            # NOTE: if the process fails here, this is likely an issue with the
            # calculation of the proxy relative values (i.e. they do not all sum to 1
            # by the geo/time level)
            if not allocation_qc_pass:
                break

            try:
                # STEP X: GRID EMISSIONS
                # turn the vector proxy into a grid
                gridded_results = grid_allocated_emissions(
                    allocation_gdf
                )
            except Exception as e:
                logging.critical(f"{row.emi_id}, {row.proxy_id} gridding failed {e}")
                break
            # STEP X: QC GRIDDED EMISSIONS -----------------------------------------
            # TODO: update this to generalize geo/time levels QC instead of just year.
            # NOTE: if QC fails here, I'm not yet
            raster_qc_pass = QC_emi_raster_sums(
                gridded_results, emi_df
            )
            source_group_dict[row.emi_id]["rasters"] = gridded_results

        elif row.file_type == "netcdf":
            logging.info(f"{row.proxy_id} is a raster.")

            # read the proxy file
            proxy_ds = xr.open_dataset(proxy_input_path).rename({"geoid": geo_col})

            # if the emi is month and the proxy is annual, we expand the dimensions of
            # the proxy, repeating the year values for every month in the year
            # we stack the year/month dimensions into a single year_month so that
            # it aligns with the emissions data as a time x X x Y array.
            if row.emi_time_step == "monthly" and row.proxy_time_step == "annual":
                proxy_ds = proxy_ds.expand_dims(
                    dim={"month": np.arange(1, 13)}, axis=0
                ).stack({"year_month": ["year", "month"]})
            proxy_ds

            proxy_fips = np.unique(proxy_ds.fips.values)
            emi_fips = np.unique(emi_df.fips.values)
            missing_fips = set(emi_fips) - set(proxy_fips)
            if len(missing_fips) != 0:
                logging.critical(f"{row.proxy_id} missing fips: {missing_fips}")

            match row.emi_geo_level:
                case "state":
                    admin_gdf = state_gdf
                case "county":
                    admin_gdf = county_gdf
            # NOTE: this is slow.
            emi_rasters_3d = make_emi_grid(emi_df, admin_gdf, time_col)

            # assign the emissions array to the proxy dataset
            proxy_ds["emissions"] = (proxy_ds.dims, emi_rasters_3d)
            # # look to make sure the emissions loaded in the correct orientation
            # proxy_ds["emissions"].sel(year_month=(2020, 1)).where(lambda x: x > 0).plot(
            #     cmap="hot"
            # )
            # plt.show()
            # calculate the emissions for the proxy array by multiplying the proxy by
            # the gridded emissions data
            proxy_ds["results"] = (
                proxy_ds[row.proxy_rel_emi_col] * proxy_ds["emissions"]
            )

            yearly_results = proxy_ds["results"].groupby("year").sum()
            raster_qc_pass = QC_emi_raster_sums(yearly_results, emi_df)
            if not raster_qc_pass:
                break
            source_group_dict[row.emi_id]["rasters"] = yearly_results

            # # plot the results
            # proxy_ds["results"].sel(year_month=(2020, 1)).where(lambda x: x > 0).plot(
            #     cmap="Spectral"
            # )
            # plt.show()

    # STEP 5.2: COMBINE SUBSOURCE RASTERS TOGETHER ------------------------------
    # if all the emi/proxy pairs were successful, we can combine the rasters and
    # calculate the flux.

    gridded_emis = list(source_group_dict.keys())
    expected_emis = data.emi_id.unique().tolist()
    all_emis_created = all([x in gridded_emis for x in expected_emis])
    if [x for x in expected_emis if x not in gridded_emis]:
        missing_emis = ", ".join([x for x in expected_emis if x not in gridded_emis])
        logging.critical(
            f"{source_name} has failed\n" f"missing emi/proxy pairs for {missing_emis}"
        )
        continue

    rasters_exist = [
        "rasters" in source_group_dict[emi_name].keys()
        for emi_name in source_group_dict.keys()
    ]
    if not all(rasters_exist):
        logging.critical(
            f"{source_name} has failed\n" f"No raster data for {source_name}"
        )
        continue

    raster_list = [
        source_group_dict[emi_name]["rasters"] for emi_name in source_group_dict.keys()
    ]
    ch4_kt_result_rasters = combine_gridded_emissions(raster_list)
    ch4_flux_result_rasters = calculate_flux(ch4_kt_result_rasters)

    # STEP 5.2: QC FLUX AGAINST V2 ----------------------------------------------
    # TODO: the one thing the automated emi/proxy mapping script does not do is go find
    # the name of the v2 file. I don't think there is any way around doing that manually
    # from v2 to v3. It would be possible in the future since there will be closer
    # alignment in the data structure between the two versions.
    # flux_emi_qc_df = QC_flux_emis(ch4_flux_result_rasters, source_name, v2_name=v2_name)
    # flux_emi_qc_df
    # STEP 6: SAVE THE FILES ----------------------------------------------------
    write_tif_output(ch4_kt_result_rasters, tif_kt_output_path)
    write_tif_output(ch4_flux_result_rasters, tif_flux_output_path)
    write_ncdf_output(
        ch4_flux_result_rasters,
        nc_flux_output_path,
        netcdf_title,
        netcdf_description,
    )

    # STEP 7: PLOT THE RESULTS AND DIFFERENCE, SAVE FIGURES TO FILES ------------
    plot_annual_raster_data(ch4_flux_result_rasters, source_name)
    plot_raster_data_difference(ch4_flux_result_rasters, source_name)

    logging.info(f"Finished {source_name}\n\n")
    # %%
    i = 0
    for id, values in gridding_params.items():
        i += len([x for x in values["proxy_input_paths"] if x.suffix == ".parquet"])
    i


# %%
_id = "2C1_iron_and_steel"
# _id = "1B2bii_ng_production"  # this has monthly proxy and annual emi
# _id = "3F4_fbar"
# _id = "3A_enteric_fermentation"  # monthly emi to annual gridded proxy
kwargs = gridding_params[_id]

(
    source_name,
    data,
    nc_flux_output_path,
    nc_kt_output_path,
    tif_flux_output_path,
    tif_kt_output_path,
    netcdf_title,
    netcdf_description,
) = kwargs.values()

for row in data.itertuples(index=False):
    print(row.emi_time_step)
    print(row.emi_id)
    print(row.proxy_time_step)
    print(row.proxy_id)
    print()
    emi_input_path = list(emi_data_dir_path.glob(f"{row.emi_id}.csv"))[0]
    proxy_input_path = list(proxy_data_dir_path.glob(f"{row.proxy_id}.*"))[0]
# %%
