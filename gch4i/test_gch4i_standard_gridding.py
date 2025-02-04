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
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pyarrow import parquet  # noqa
import osgeo  # noqa
import geopandas as gpd
import numpy as np
import pandas as pd
from pytask import Product, mark, task
from tqdm.auto import tqdm, trange

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    global_data_dir_path,
    proxy_data_dir_path,
    tmp_data_dir_path,
)
from gch4i.utils import (
    QC_emi_raster_sums,
    QC_flux_emis,
    QC_proxy_allocation,
    allocate_emissions_to_proxy,
    calculate_flux,
    combine_gridded_emissions,
    grid_allocated_emissions,
    plot_annual_raster_data,
    plot_raster_data_difference,
    write_ncdf_output,
    write_tif_output,
)

# from pytask import Product, task


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

        emi_list = [list(emi_data_dir_path.glob(f"{x}.*"))[0] for x in data.emi_id]
        proxy_file_list = [
            list(proxy_data_dir_path.glob(f"{x}.*"))[0] for x in data.proxy_id
        ]

        emi_time_cols = []
        proxy_time_cols = []
        for row in data.itertuples():
            if row.emi_has_year_col & row.emi_has_month_col:
                emi_time_cols.append(["year", "month"])
            elif row.emi_has_year_col:
                emi_time_cols.append(["year"])
            if row.proxy_has_year_col & row.proxy_has_month_col:
                proxy_time_cols.append(["year", "month"])
            elif row.proxy_has_year_col:
                proxy_time_cols.append(["year"])
        for emi_timestep, proxy_timestep in zip(emi_time_cols, proxy_time_cols):
            if not emi_timestep == proxy_timestep:
                logging.warning(
                    f"{gch4i_name}: issue with timestep alignment "
                    f"{row.emi_id}, {row.proxy_id}"
                )

        gridding_params[gch4i_name] = dict(
            source_name=gch4i_name,
            # v2_name=data.v2_name.iloc[0],
            emi_input_paths=emi_list,
            emi_has_state_col=data.emi_has_state_col.to_list(),
            emi_timestep_list=emi_time_cols,
            proxy_input_paths=proxy_file_list,
            proxy_timtestep_list=proxy_time_cols,
            proxy_has_prop_list=data.proxy_has_rel_emi_col.to_list(),
            proxy_prop_col_list=data.proxy_rel_emi_col.to_list(),
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
mapping_path = V3_DATA_PATH.parents[1] / "emi_proxy_mapping_output.csv"
mapping_df = pd.read_csv(mapping_path)

# currently we are missing some proxies, so we drop any gridding group that does not
# have all the proxies ready.
groups_ready_for_gridding = mapping_df.groupby("gch4i_name")["proxy_has_file"].all()
groups_ready_for_gridding = groups_ready_for_gridding[groups_ready_for_gridding]
# filter the mapping df to only include the groups that are ready for gridding.
ready_mapping_df = mapping_df[
    mapping_df.gch4i_name.isin(groups_ready_for_gridding.index)
]
ready_mapping_df

# %%

# we need this as a reference file to filter out the states that are not in the gridding
# region of lower 48 + DC.
state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
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


gridding_params = create_gridding_params(ready_mapping_df)
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
for i, (_id, kwargs) in tqdm(enumerate(gridding_params.items()), total=N_SOURCES_READY):

    (
        source_name,
        emi_input_paths,
        emi_has_state_col,
        emi_timestep_list,
        proxy_input_paths,
        proxy_timtestep_list,
        proxy_has_prop_list,
        proxy_prop_col_list,
        nc_flux_output_path,
        nc_kt_output_path,
        tif_flux_output_path,
        tif_kt_output_path,
        netcdf_title,
        netcdf_description,
    ) = kwargs.values()

    logging.info(f"\n\n#{i}/{N_SOURCES_READY} Starting {source_name}")

    file_status = [
        x.exists()
        for x in [
            nc_flux_output_path,
            # nc_kt_output_path,
            tif_flux_output_path,
            tif_kt_output_path,
        ]
    ]
    if all(file_status):
        logging.info(f"{source_name} already gridded.\n")
        continue

    # get the iterator for the emi/proxy pairs.
    emi_proxy_gridding_iter = zip(
        emi_input_paths,
        emi_has_state_col,
        emi_timestep_list,
        proxy_input_paths,
        proxy_timtestep_list,
        proxy_has_prop_list,
        proxy_prop_col_list,
    )

    # a dictionary to hold the gridding group gridding results for all emi/proxy pairs
    emi_dict = {}
    # for each emi/proxy pair in the gridding group
    for (
        emi_input_path,
        emi_has_state_col,
        emi_timestep,
        proxy_input_path,
        proxy_timestep,
        proxy_has_prop,
        proxy_prop_col,
    ) in emi_proxy_gridding_iter:
        # break
        emi_name = emi_input_path.stem
        proxy_name = proxy_input_path.stem
        logging.info(f"Processing {emi_name}, {proxy_name}")
        # STEP 2: Read In EPA State/year GHGI Emissions
        emi_dict[emi_name] = {}
        # read the emi file and filter out the states that are not in the lower 48 + DC
        # and remove any records with 0 emissions.

        if emi_has_state_col:
            emi_cols = emi_timestep + ["state_code", "ghgi_ch4_kt"]
            emi_df = pd.read_csv(emi_input_path, usecols=emi_cols).query(
                "(state_code.isin(@state_gdf.state_code)) & (ghgi_ch4_kt > 0)"
            )
        else:
            emi_cols = emi_timestep + ["ghgi_ch4_kt"]
            emi_df = pd.read_csv(emi_input_path, usecols=emi_cols)

        # STEP 3: GET AND FORMAT PROXY DATA
        # we apply a different set of functions depending on the proxy type.
        # if vector type
        if proxy_input_path.suffix == ".parquet":

            # read the file, catch any errors and log them.
            try:
                proxy_gdf = gpd.read_parquet(proxy_input_path).reset_index()
            except Exception as e:
                logging.critical(f"Error reading {proxy_input_path.name}: {e}")
                continue

            # if the emi_time and proxy_time are the same, we can allocate the emissions
            # directly to the proxies since everything aligns.
            # TODO: add an arg to the allocate_emissions_to_proxy function to take in
            # the timestep columns and use them as groupers instead of hardcoding year.
            # TODO: add emi_timestep == ["year"] | ["year", "month"] to the if statement
            if (
                (emi_timestep == ["year"])
                & (emi_timestep == proxy_timestep)
                & emi_has_state_col
            ):
                # STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES
                # embedded in this function is the check that the proxy has the same
                # year as the emissions and same set of states. It will log when that is
                # not the case since it is a critical error that needs to be corrected.

                # check if all the proxy state / time columns are in the emissions data
                # NOTE: this check happens again later, but at a siginificant time cost
                # if the data are large. It is here to catch the error early and break
                # the loop with critical message.
                proxy_unique = (
                    proxy_gdf[["state_code", "year"]]
                    .drop_duplicates()
                    .assign(proxy=1)
                    .set_index(["state_code", "year"])
                    .sort_index()
                )
                emi_unique = (
                    emi_df[["state_code", "year"]]
                    .drop_duplicates()
                    .assign(emi=1)
                    .set_index(["state_code", "year"])
                    .sort_index()
                )
                match_check = emi_unique.join(proxy_unique, how="left")

                if match_check["proxy"].isna().any():
                    logging.critical(
                        f"QC FAILED: {emi_name}, {proxy_name} "
                        "proxy state/year columns do not match emissions\n"
                        f"{match_check[match_check["proxy"].isna()].to_string().replace("\n", "\n\t")}"
                        "\n"
                    )
                    break

                emi_dict[emi_name]["allocated"] = allocate_emissions_to_proxy(
                    proxy_gdf,
                    emi_df,
                    proxy_has_year=True,
                    use_proportional=proxy_has_prop,
                    proportional_col_name=proxy_prop_col,
                )
                # STEP X: QC ALLOCATION
                # check that the allocated emissions sum to the original emissions
                emi_dict[emi_name]["allocation_qc"] = QC_proxy_allocation(
                    emi_dict[emi_name]["allocated"], emi_df
                )
                # if the allocation failed, log the error and break the loop.
                if not emi_dict[emi_name]["allocation_qc"].isclose.all():
                    logging.critical(
                        f"QC FAIL: {emi_name}, {proxy_name}." f"allocation failed."
                    )
                    break

                try:
                    # STEP X: GRID EMISSIONS
                    emi_dict[emi_name]["rasters"] = grid_allocated_emissions(
                        emi_dict[emi_name]["allocated"]
                    )
                except Exception as e:
                    logging.critical(f"{emi_name}, {proxy_name} gridding failed {e}")
                    # break out of outer loop if gridding fails.
                    break

                # STEP X: QC GRIDDED EMISSIONS ---------------------------------------------
                # TODO: update this to do state+year QC instead of just year.
                emi_dict[emi_name]["raster_qc"] = QC_emi_raster_sums(
                    emi_dict[emi_name]["rasters"], emi_df
                )
            else:
                logging.warning(
                    f"{emi_name}, {proxy_name} not implemented yet\n"
                    f"emi has state: {emi_has_state_col}\n"
                    f"time: {emi_timestep}, {proxy_timestep}\n"
                    f"state: {emi_has_state_col}\n"
                    f"rel_emis: {proxy_has_prop}, {proxy_prop_col}\n"
                )
        # raster proxies will probably need a different process from read to
        # allocate. Then QC_emi_raster_sums and beyond would be the same.
        # if proxy_type == "raster":
        #     pass
        if proxy_input_path.suffix == ".nc":
            logging.warning("raster proxies not implemented yet")
            continue

    # STEP 5.2: COMBINE SUBSOURCE RASTERS TOGETHER ------------------------------
    # if all the emi/proxy pairs were successful, we can combine the rasters and
    # calculate the flux.

    gridded_emis = list(emi_dict.keys())
    if len(gridded_emis) == 0:
        # logging.critical(f"No successful emi/proxy pairs for {source_name}")
        continue
    if len(gridded_emis) != len(emi_input_paths):
        # logging.critical(f"missing emi/proxy pair(s) for {source_name}")
        continue

    rasters_exist = [
        "rasters" in emi_dict[emi_name].keys() for emi_name in emi_dict.keys()
    ]
    if not all(rasters_exist):
        # logging.critical(f"No raster data for {source_name}")
        continue

    raster_list = [emi_dict[emi_name]["rasters"] for emi_name in emi_dict.keys()]
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
        i += len([x for x in values["proxy_input_paths"] if x.suffix==".parquet"])
    i
# %%

# %%
