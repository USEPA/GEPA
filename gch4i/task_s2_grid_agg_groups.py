import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from IPython.display import display
from tqdm.auto import tqdm

from gch4i.config import V3_DATA_PATH, gridded_output_dir, tmp_data_dir_path, years
from gch4i.create_emi_proxy_mapping import get_gridding_mapping_df
from gch4i.utils import (
    QC_flux_emis,
    calculate_flux,
    plot_annual_raster_data,
    plot_raster_data_difference,
    write_ncdf_output,
    write_tif_output,
)


def get_status_table(status_db_path):
    # get a numan readable version of the status database
    conn = sqlite3.connect(status_db_path)
    status_df = pd.read_sql_query("SELECT * FROM gridding_status", conn)
    conn.close()
    return status_df


working_dir = V3_DATA_PATH.parents[0] / "gridding_log_and_qc"
status_db_path: Path = working_dir / "gridding_status.db"
v2_data_path: Path = V3_DATA_PATH.parents[1] / "v2_v3_comparison_crosswalk.csv"
mapping_file_path: Path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

mapping_df = get_gridding_mapping_df(mapping_file_path)
v2_df = pd.read_csv(v2_data_path).rename(columns={"v3_gch4i_name": "gch4i_name"})
status_df = get_status_table(status_db_path, working_dir)
status_df = status_df.merge(mapping_df, on=["gch4i_name", "emi_id", "proxy_id"])
print("percent of emi/proxy pairs by status")
display(status_df["status"].value_counts(normalize=True).multiply(100).round(2))




# %%
group_ready_status = (
    status_df.groupby("gch4i_name")["status"]
    .apply(lambda x: x.eq("complete").all())
    .to_frame()
)
print("percent of gridding groups ready")
display(
    group_ready_status["status"].value_counts(normalize=True).multiply(100).round(2)
)

ready_groups = group_ready_status[group_ready_status["status"] == True].join(
    v2_df.set_index("gch4i_name")
)
ready_groups
# %%
conn = sqlite3.connect(status_db_path)
cursor = conn.cursor()

for g_name, status in tqdm(
    ready_groups.iterrows(), total=len(ready_groups), desc="gridding groups"
):
    group_name = g_name.lower()
    v2_name = status.v2_key
    tif_flux_output_path = tmp_data_dir_path / f"{group_name}_ch4_emi_flux.tif"
    tif_kt_output_path = tmp_data_dir_path / f"{group_name}_ch4_kt_per_year.tif"
    nc_flux_output_path = tmp_data_dir_path / f"{group_name}_ch4_emi_flux.nc"

    # if (
    #     tif_flux_output_path.exists()
    #     and tif_kt_output_path.exists()
    #     and nc_flux_output_path.exists()
    # ):
    #     logging.info(f"{group_name} output files already exist.\n")
    #     continue

    IPCC_ID, SOURCE_NAME = group_name.split("_", maxsplit=1)

    netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
    netcdf_description = (
        f"Gridded EPA Inventory - {group_name} - "
        f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
    )
    # netcdf_title = f"CH4 emissions from {source_name} gridded to 1km x 1km"
    all_raster_list = list(gridded_output_dir.glob(f"{group_name}*.tif"))
    annual_raster_list = [
        raster for raster in all_raster_list if "monthly" not in raster.name
    ]
    # this is the expected number of rasters for this group
    group_mapping_count = mapping_df[mapping_df["gch4i_name"] == g_name].shape[0]

    if annual_raster_list:
        print(f"{group_name} has annual rasters. count: {len(annual_raster_list)}")
        # Check the count against the mapping_df count for this gridding group
        if len(annual_raster_list) != group_mapping_count:
            # logging.critical(f"{group_name} has {len(annual_raster_list)} rasters,
            # but mapping_df has {group_mapping_count} entries.")
            raise ValueError("Mismatch in raster count and mapping_df count.")
        annual_arr_list = []
        for raster_path in annual_raster_list:
            with rasterio.open(raster_path) as src:
                annual_arr_list.append(src.read())

        if len(annual_arr_list) > 1:
            annual_summed_arr = np.sum(annual_arr_list, axis=0)
        else:
            annual_summed_arr = annual_arr_list[0]
        annual_summed_arr.shape

        annual_summed_da = xr.DataArray(
            annual_summed_arr,
            dims=["time", "y", "x"],
            coords={
                "time": years,
                "y": np.arange(annual_summed_arr.shape[1]),
                "x": np.arange(annual_summed_arr.shape[2]),
            },
            name=group_name,
            attrs={
                "units": "kt",
                "long_name": f"CH4 emissions from {group_name} gridded to 1km x 1km",
                "source": group_name,
                "description": f"CH4 emissions from {group_name} gridded to 1km x 1km",
            },
        )

        # ch4_kt_result_da = combine_gridded_emissions(annual_raster_list)
        write_tif_output(annual_summed_da, tif_kt_output_path)

        # STEP 5.2: QC FLUX AGAINST V2 ----------------------------------------------
        ch4_flux_result_da = calculate_flux(annual_summed_da)
        if pd.isna(v2_name):
            # logging.warning(f"Skipping {group_name} because v2_name is NaN.")
            print("no v2 version")
        else:
            flux_emi_qc_df = QC_flux_emis(
                ch4_flux_result_da, group_name, v2_name=v2_name
            )
            flux_emi_qc_df
        plot_annual_raster_data(ch4_flux_result_da, group_name)
        plot_raster_data_difference(ch4_flux_result_da, group_name)
        # STEP 6: SAVE THE FILES ----------------------------------------------------
        write_tif_output(ch4_flux_result_da, tif_flux_output_path)
        write_ncdf_output(
            ch4_flux_result_da,
            nc_flux_output_path,
            netcdf_title,
            netcdf_description,
        )

    # monthly_raster_list = [
    #     raster for raster in all_raster_list if "monthly" in raster.name
    # ]
    # if monthly_raster_list:
    #     print("\t\tmonthly", len(monthly_raster_list))
    #     if len(monthly_raster_list) != group_mapping_count:
    #         # logging.critical(f"{group_name} has {len(monthly_raster_list)} rasters,
    #         # but mapping_df has {group_mapping_count} entries.")
    #         # raise ValueError("Mismatch in raster count and mapping_df count.")
    #         print(f"we don't have all the monthly rasters we need for {group_name}")
    #         continue
    #     else:
    #         monthly_arr_list = []
    #         for raster_path in monthly_raster_list:
    #             with rasterio.open(raster_path) as src:
    #                 monthly_arr_list.append(src.read())

    #         if len(monthly_arr_list) > 1:
    #             monthly_summed_arr = np.sum(monthly_arr_list, axis=0)
    #         else:
    #             monthly_summed_arr = monthly_arr_list[0]
    #         monthly_summed_arr.shape
    #         year_months = (
    #             pd.date_range(
    #                 start=f"{years[0]}-01",
    #                 end=f"{years[-1] + 1}-01",
    #                 freq="ME",
    #                 inclusive="both",
    #             )
    #             .strftime("%Y-%m")
    #             .tolist()
    #         )
    #         monthly_summed_da = xr.DataArray(
    #             monthly_summed_arr,
    #             dims=["time", "y", "x"],
    #             coords={
    #                 "time": year_months,
    #                 "y": np.arange(monthly_summed_arr.shape[1]),
    #                 "x": np.arange(monthly_summed_arr.shape[2]),
    #             },
    #             name=group_name,
    #             attrs={
    #                 "units": "kt",
    #                 "long_name": f"CH4 emissions from {group_name} gridded to 1km x 1km",
    #                 "source": group_name,
    #                 "description": f"CH4 emissions from {group_name} gridded to 1km x 1km",
    #             },
    #         )

    # logging.info(f"Finished {group_name}\n\n")
# %%
