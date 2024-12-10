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
Notes:                  - 
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------

from pathlib import Path
from typing import Annotated

# for testing/development
# %load_ext autoreload
# %autoreload 2

import osgeo  # noqa
import geopandas as gpd
import numpy as np
import pandas as pd


from pytask import Product, mark, task

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    proxy_data_dir_path,
    tmp_data_dir_path,
)
from gch4i.utils import (
    QC_emi_raster_sums,
    QC_proxy_allocation,
    QC_flux_emis,
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

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)


# %% STEP 1. Load GHGI-Proxy Mapping Files


proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_mapping = (
    pd.read_excel(proxy_file_path, sheet_name="gridding_flag_info")
    # .query(f"gch4i_name == '{GCH4I_NAME}'")
    # .drop_duplicates(subset=["gch4i_name", "emi", "proxy"])
)
proxy_mapping


# this would be a list of all the gridding souces that this process could be applied to.
# these would be those that follow the standard process
# NOTE: these are just example names listed for testing / demo.
# gridding_list = ["1B1a_abandoned_coal", "3F4_fbar", "1B1a_coal_mining_underground"]
"""
things we need to stick in the spreadsheet:
 - file extentions for emis and proxies
 - v2 name
 - proxy_has_year=True,
 - use_proportional=True,
 - proportional_col_name="emis_mmcfd",
 - proxy data type
"""

gridding_params = {}
for gch4i_name, data in proxy_mapping.groupby("gch4i_name"):
    # if gch4i_name not in gridding_list:
    # continue

    GRIDDING_NAME = data.gch4i_name.iloc[0].lower()
    # v2_name = data.v2_name.iloc[0]
    IPCC_ID, SOURCE_NAME = gch4i_name.split("_", maxsplit=1)

    # FULL_NAME = "_".join([IPCC_ID, GRIDDING_NAME, SOURCE_NAME]).replace(" ", "_")

    netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
    netcdf_description = (
        f"Gridded EPA Inventory - {GRIDDING_NAME} - "
        f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
    )

    emi_list = [(emi_data_dir_path / x).with_suffix(".csv") for x in data.emi_id]
    proxy_list = [
        (proxy_data_dir_path / x).with_suffix(".parquet") for x in data.proxy_id
    ]
    proxy_type_list = data.proxy_type.to_list()
    use_prop_col_list = data.proxy_use_prop.to_list()
    prop_col_name_list = data.proxy_prop_col_name.to_list()
    proxy_timestep_list = data.proxy_timestep.to_list()
    emi_timestep_list = data.emi_timestep.to_list()

    gridding_params[gch4i_name] = dict(
        source_name=gch4i_name,
        v2_name=data.v2_name.iloc[0],
        emi_input_paths=emi_list,
        emi_timestep_list=emi_timestep_list,
        proxy_input_paths=proxy_list,
        proxy_type_list=proxy_type_list,
        proxy_timtestep=proxy_timestep_list,
        proxy_has_prop_list=use_prop_col_list,
        proxy_prop_col_list=prop_col_name_list,
        nc_flux_output_path=tmp_data_dir_path / f"{GRIDDING_NAME}_ch4_emi_flux.nc",
        nc_kt_output_path=tmp_data_dir_path / f"{GRIDDING_NAME}_ch4_kt_per_year.nc",
        tif_flux_output_path = tmp_data_dir_path / f"{GRIDDING_NAME}_ch4_emi_flux.tif",
        tif_kt_output_path = tmp_data_dir_path / f"{GRIDDING_NAME}_ch4_kt_per_year.tif",
        # "annual_plot_output_path = "",
        # "diff_plot_output_path = "",
    )
gridding_params

(
    source_name,
    v2_name,
    emi_input_paths,
    emi_timestep_list,
    proxy_input_paths,
    proxy_type_list,
    proxy_timtestep_list,
    proxy_has_prop_list,
    proxy_prop_col_list,
    nc_flux_output_path,
    nc_kt_output_path,
) = gridding_params["1B1a_abandoned_coal"].values()


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


gridding_iter = zip(
    emi_input_paths,
    emi_timestep_list,
    proxy_input_paths,
    proxy_type_list,
    proxy_timtestep_list,
    proxy_has_prop_list,
    proxy_prop_col_list,
)

# %%
emi_dict = {}
for (
    emi_input_path,
    emi_timestep,
    proxy_input_path,
    proxy_type,
    proxy_timestep,
    proxy_has_prop,
    proxy_prop_col_name,
) in gridding_iter:
    emi_name = emi_input_path.stem

    # STEP 2: Read In EPA State GHGI Emissions by Year
    # EEM: question -- can we create a script that we can run separately to run
    # all the get_emi and get_proxy functions? Then we can include a comment in
    # this script that states that those functions need to be run first Also see
    # comments on the emissions script. The emission values need to be corrected
    emi_dict[emi_name] = {}
    emi_dict[emi_name]["emi"] = pd.read_csv(emi_input_path)

    # STEP 3: GET AND FORMAT PROXY DATA
    if proxy_type == "vector":

        emi_dict[emi_name]["proxy"] = gpd.read_parquet(proxy_input_path)

        if emi_timestep == proxy_timestep:
            # STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES
            emi_dict[emi_name]["allocated"] = allocate_emissions_to_proxy(
                emi_dict[emi_name]["proxy"],
                emi_dict[emi_name]["emi"],
                proxy_has_year=True,
                use_proportional=proxy_has_prop,
                proportional_col_name=proxy_prop_col_name,
            )
        # STEP X: QC ALLOCATION
        emi_dict[emi_name]["allocation_qc"] = QC_proxy_allocation(
            emi_dict[emi_name]["allocated"], emi_dict[emi_name]["emi"]
        )
        # STEP X: GRID EMISSIONS
        emi_dict[emi_name]["rasters"] = grid_allocated_emissions(
            emi_dict[emi_name]["allocated"]
        )

    # raster proxies will probably need a different process from read to
    # allocate. Then QC_emi_raster_sums and beyond would be the same.
    # if proxy_type == "raster":
    #     pass
    elif proxy_type == "raster":
        pass

    # STEP X: QC GRIDDED EMISSIONS ---------------------------------------------
    emi_dict[emi_name]["raster_qc"] = QC_emi_raster_sums(
        emi_dict[emi_name]["rasters"], emi_dict[emi_name]["emi"]
    )

# %% STEP 5.2: COMBINE SUBSOURCE RASTERS TOGETHER ------------------------------
raster_list = [emi_dict[emi_name]["rasters"] for emi_name in emi_dict.keys()]
ch4_kt_result_rasters = combine_gridded_emissions(raster_list)
ch4_flux_result_rasters = calculate_flux(ch4_kt_result_rasters)

# %% STEP 5.2: QC FLUX AGAINST V2 ----------------------------------------------
# EEM: comment - do we need to import these functions if there were already
# imported in the utils.py script?
flux_emi_qc_df = QC_flux_emis(ch4_flux_result_rasters, source_name, v2_name=v2_name)
flux_emi_qc_df
# %% STEP 6: SAVE THE FILES ----------------------------------------------------
write_tif_output(ch4_kt_result_rasters, nc_kt_output_path)
write_tif_output(ch4_flux_result_rasters, nc_flux_output_path)
write_ncdf_output(
    ch4_flux_result_rasters,
    nc_flux_output_path,
    netcdf_title,
    netcdf_description,
)

# %% STEP 7: PLOT THE RESULTS AND DIFFERENCE, SAVE FIGURES TO FILES ------------
plot_annual_raster_data(ch4_flux_result_rasters, source_name)
plot_raster_data_difference(ch4_flux_result_rasters, source_name)


# %%
