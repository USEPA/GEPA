"""
Name:                   test_gch4i_standard_gridding.py
Date Last Modified:     2024-06-07
Authors Name:           N. Kruskamp, H. Lohman (RTI International), Erin McDuffie
                        (EPA/OAP)
Purpose:                This is a template framework for a gridder that applies
                        to every gch4i source that uses a one-to-one emissions to proxy
                        allocation and does not include things like monthly emissions,
                        intermediate proxies, etc. The user were specify the list of
                        gch4i gridding sources and the gridder would be applied to each
                        one of them, pairing the emission and proxies and other
                        argumentes needed from the proxy mapping spreadsheet.
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


import matplotlib.pyplot as plt

# %%
import osgeo
import rasterio
import rioxarray
import xarray as xr


def qc_flux_emis(data: dict, v2_name: str):

    v2_data_paths = V3_DATA_PATH.glob("Gridded_GHGI_Methane_v2_*.nc")
    v2_data_dict = {}
    for in_path in v2_data_paths:
        v2_year = int(in_path.stem.split("_")[-1])
        v2_data = rioxarray.open_rasterio(in_path, variable=v2_name)[
            v2_name
        ].values.squeeze(axis=0)
        v2_data = np.where(v2_data == 0, np.nan, v2_data)
        v2_data_dict[v2_year] = v2_data

    result_list = []
    for year in data.keys():
        if year in v2_data_dict.keys():
            v3_data = np.where(data[year] == 0, np.nan, data[year])
            yearly_dif = data[year] - v2_data_dict[year]
            v2_sum = np.nansum(v2_data)
            v3_sum = np.nansum(v3_data)
            print(f"v2 sum: {v2_sum}, v3 sum: {v3_sum}")
            result_list.append(
                pd.DataFrame(yearly_dif.ravel())
                .dropna(how="all", axis=1)
                .describe()
                .rename(columns={0: year})
            )

        # TODO: save out a map if differences
        # compare sums of arrays by year

        fig, (ax1, ax2) = plt.subplots(2)
        fig.tight_layout()
        ax1.hist(v2_data.ravel(), bins=100)
        ax2.hist(v3_data.ravel() / 1000, bins=100)
        plt.show()

        np.nanmax(v3_data)

        np.nanmax(v2_data)

    result_df = pd.concat(result_list, axis=1)
    return result_df


# %% STEP 1. Load GHGI-Proxy Mapping Files


proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_mapping = (
    pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping")
    # .query(f"gch4i_name == '{GCH4I_NAME}'")
    .drop_duplicates(subset=["gch4i_name", "emi", "proxy"])
)
proxy_mapping

"""
things we need to stick in the spreadsheet:
 - file extentions for emis and proxies
 - v2 name
 - proxy_has_year=True,
 - use_proportional=True,
 - proportional_col_name="emis_mmcfd",
 - proxy data type
"""

# %%
# this would be a list of all the gridding souces that this process could be applied to.
# these would be those that follow the standard process
# NOTE: these are just example names listed for testing / demo.
gridding_list = ["1B1a_abandoned_coal", "3F4_fbar", "1B1a_coal_mining_underground"]

gridding_params = {}
for gch4i_name, data in proxy_mapping.groupby("gch4i_name"):
    if gch4i_name not in gridding_list:
        continue

    SECTOR_NAME = data.Sector.iloc[0].lower()
    # v2_name = data.v2_name.iloc[0]
    IPCC_ID, SOURCE_NAME = gch4i_name.split("_", maxsplit=1)

    FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

    netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
    netcdf_description = (
        f"Gridded EPA Inventory - {SECTOR_NAME} - "
        f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
    )

    gridding_params[gch4i_name] = {
        "source_name": gch4i_name,
        # "v2_name": v2_name,
        "emi_inputs": [(emi_data_dir_path / x).with_suffix(".csv") for x in data.emi],
        "proxy_inputs": [
            (proxy_data_dir_path / x).with_suffix(".parquet") for x in data.proxy
        ],
        # "proxy_data_type": list[str] = data.proxy_data_type.to_list(), vector or raster
        # "proxy_has_year": list[bool] = data.proxy_has_year.to_list(),
        # "proxy_has_prop"; list[bool] = data.proxy_has_prop.to_list(),
        # "proxy_prop_col": list[str] = data.proxy_has_prop.to_list(),
        "nc_flux_output_path": tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux.nc",
        "nc_kt_output_path": tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year.nc",
        # "tif_flux_output_path": tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux.tif",
        # "tif_kt_output_path": tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year.tif",
        # "annual_plot_output_path": "",
        # "diff_plot_output_path": "",
    }
gridding_params


# %% READ IN EPA DATA, ALLOCATE, GRID, AND QC ----------------------------------

for _id, kwargs in gridding_params.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_gridding(
        source_name: str,
        v2_name: str,
        emi_inputs: list[Path],
        proxy_inputs: list[Path],
        nc_flux_output_path: Annotated[Path, Product],
        nc_kt_output_path: Annotated[Path, Product],
    ) -> None:

        emi_dict = {}
        for emi_input_path, proxy_input_path in zip(emi_inputs, proxy_inputs):
            emi_name = emi_input_path.stem

            # STEP 2: Read In EPA State GHGI Emissions by Year
            # EEM: question -- can we create a script that we can run separately to run
            # all the get_emi and get_proxy functions? Then we can include a comment in
            # this script that states that those functions need to be run first Also see
            # comments on the emissions script. The emission values need to be corrected
            emi_dict[emi_name] = {}
            emi_dict[emi_name]["emi"] = pd.read_csv(emi_input_path)

            # STEP 3: GET AND FORMAT PROXY DATA ----------------------------------------
            # if proxy_type == "vector":
            #     pass

            emi_dict[emi_name]["proxy"] = gpd.read_parquet(proxy_input_path)

            # STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES ------------------
            emi_dict[emi_name]["allocated"] = allocate_emissions_to_proxy(
                emi_dict[emi_name]["proxy"],
                emi_dict[emi_name]["emi"],
                proxy_has_year=True,
                use_proportional=True,
                proportional_col_name="emis_mmcfd",
            )
            # STEP X: QC ALLOCATION ----------------------------------------------------
            emi_dict[emi_name]["allocation_qc"] = QC_proxy_allocation(
                emi_dict[emi_name]["allocated"], emi_dict[emi_name]["emi"]
            )
            # STEP X: GRID EMISSIONS ---------------------------------------------------
            emi_dict[emi_name]["rasters"] = grid_allocated_emissions(
                emi_dict[emi_name]["allocated"]
            )

            # raster proxies will probably need a different process from read to
            # allocate. Then QC_emi_raster_sums and beyond would be the same.
            # if proxy_type == "raster":
            #     pass

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
        flux_emi_qc_df = qc_flux_emis(ch4_flux_result_rasters, v2_name=v2_name)
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
