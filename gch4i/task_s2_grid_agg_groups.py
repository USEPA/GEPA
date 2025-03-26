# %%
# %load_ext autoreload
# %autoreload 2
# %%

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from IPython.display import display
from tqdm.auto import tqdm
import geopandas as gpd

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    prelim_gridded_dir,
    years,
    global_data_dir_path,
)
from gch4i.create_emi_proxy_mapping import get_gridding_mapping_df
from gch4i.utils import (
    GEPA_spatial_profile,
    QC_flux_emis,
    calculate_flux,
    plot_annual_raster_data,
    plot_raster_data_difference,
    write_ncdf_output,
    write_tif_output,
)

# %%
relative_tolerance = 0.01
gepa_profile = GEPA_spatial_profile()


def get_status_table(status_db_path):
    # get a numan readable version of the status database
    conn = sqlite3.connect(status_db_path)
    status_df = pd.read_sql_query("SELECT * FROM gridding_status", conn)
    conn.close()
    return status_df


working_dir = V3_DATA_PATH.parents[0] / "gridding_log_and_qc"
emi_grid_qc_dir = working_dir / "emi_grid_qc"

v3_inv_v3_grid_path: Path = working_dir / "v3_inventory_vs_v3_gridded"

status_db_path: Path = working_dir / "gridding_status.db"
v2_data_path: Path = V3_DATA_PATH.parents[1] / "v2_v3_comparison_crosswalk.csv"
mapping_file_path: Path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
mapping_df = get_gridding_mapping_df(mapping_file_path)
v2_df = pd.read_csv(v2_data_path).rename(columns={"v3_gch4i_name": "gch4i_name"})
status_df = get_status_table(status_db_path)
status_df = status_df.merge(mapping_df, on=["gch4i_name", "emi_id", "proxy_id"])
print("percent of emi/proxy pairs by status")
display(status_df["status"].value_counts(normalize=True).multiply(100).round(2))

gridded_output_dir = working_dir / "gridded_output"

# %%
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

ready_groups = (
    group_ready_status[group_ready_status["status"] == True]
    .join(v2_df.set_index("gch4i_name"))
    .fillna({"v2_key": ""})
    .astype({"v2_key": str})
    # .query("gch4i_name == '4D2_land_converted_to_wetlands'")
)
ready_groups
# %%

conn = sqlite3.connect(status_db_path)
cursor = conn.cursor()

for g_name, status in tqdm(
    ready_groups.iterrows(), total=len(ready_groups), desc="gridding groups"
):
    print(f"{g_name}")
    v2_name = status.v2_key
    # if g_name == "4A1_4A2_Forest_land_remaining_forest_land":
    # break
    group_name = g_name.lower()
    tif_flux_output_path = prelim_gridded_dir / f"{group_name}_ch4_emi_flux.tif"
    tif_kt_output_path = prelim_gridded_dir / f"{group_name}_ch4_kt_per_year.tif"
    nc_flux_output_path = prelim_gridded_dir / f"{group_name}_ch4_emi_flux.nc"

    qc_files = list(emi_grid_qc_dir.glob(f"{g_name}*"))
    qc_files = [x for x in qc_files if "monthly" not in x.name]
    group_sources = mapping_df[mapping_df["gch4i_name"] == g_name]
    group_mapping_count = group_sources.shape[0]
    if len(qc_files) != group_mapping_count:
        print(
            f"{g_name} has {len(qc_files)} qc files, but should have "
            f"{group_mapping_count}"
        )
        print(qc_files)
    all_qc = pd.concat([pd.read_csv(x) for x in qc_files], axis=0)
    yearly_qc = all_qc.groupby("year").agg(
        {"ghgi_ch4_kt": "sum", "results": "sum", "isclose": "all", "diff": "sum"}
    )

    emi_results_list = []
    for row in group_sources.itertuples():
        emi_df = pd.read_csv(emi_data_dir_path / f"{row.emi_id}.csv")
        try:
            emi_df = emi_df.query(
                "(state_code.isin(@state_gdf.state_code)) & (ghgi_ch4_kt > 0)"
            )
        except:
            print("national emissions")
        emi_results_list.append(emi_df)
    emi_results_df = pd.concat(emi_results_list, axis=0)
    yearly_inv_emi_df = (
        emi_results_df.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
    )
    yearly_inv_emi_df

    all_raster_list = list(gridded_output_dir.glob(f"{group_name}*.tif"))
    annual_raster_list = [
        raster for raster in all_raster_list if "monthly" not in raster.name
    ]

    monthly_raster_list = [
        raster for raster in all_raster_list if "monthly" in raster.name
    ]
    # this is the expected number of rasters for this group
    group_sources = mapping_df[mapping_df["gch4i_name"] == g_name]
    group_mapping_count = group_sources.shape[0]

    # if monthly_raster_list:
    #     if len(monthly_raster_list) != group_mapping_count:
    #         print(
    #             f"{group_name} has {len(monthly_raster_list)} rasters, but mapping_df has {group_mapping_count} entries."
    #         )
    #     else:
    #         print(f"{group_name} has ALL monthly rasters")
    # else:
    #     print(f"{group_name} does not have monthly rasters")
    # print()
    # # %%
    if annual_raster_list:
        # print(f"{group_name} has annual rasters. count: {len(annual_raster_list)}")
        # Check the count against the mapping_df count for this gridding group
        if len(annual_raster_list) != group_mapping_count:
            # logging.critical(f"{group_name} has {len(annual_raster_list)} rasters,
            # but mapping_df has {group_mapping_count} entries.")
            raise ValueError("Mismatch in raster count and mapping_df count.")
        annual_arr_list = []
        for raster_path in annual_raster_list:
            with rasterio.open(raster_path) as src:
                arr_data = src.read()
                annual_arr_list.append(src.read())

        if len(annual_arr_list) > 1:
            annual_summed_arr = np.nansum(annual_arr_list, axis=0)
        else:
            annual_summed_arr = annual_arr_list[0]
        annual_summed_arr.shape

        gridded_sums = np.nansum(annual_summed_arr, axis=(1, 2))

        emi_check_df = yearly_inv_emi_df.assign(gridded_emissions=gridded_sums).assign(
            yearly_dif=lambda df: df["ghgi_ch4_kt"] - df["gridded_emissions"],
            rel_diff=lambda df: np.abs(
                100
                * (df["ghgi_ch4_kt"] - df["gridded_emissions"])
                / ((df["ghgi_ch4_kt"] + df["gridded_emissions"]) / 2)
            ),
            qc_pass=lambda df: (df["rel_diff"] < relative_tolerance),
            grid_pass=lambda df: np.isclose(
                df["ghgi_ch4_kt"], df["gridded_emissions"], atol=0.0, rtol=0.0001
            ),
        )
        emi_check_df = emi_check_df.merge(yearly_qc, on="year").assign(
            emi_eq=lambda df: np.isclose(
                df["ghgi_ch4_kt_x"], df["ghgi_ch4_kt_y"], atol=0.0, rtol=0.0001
            ),
            grid_eq=lambda df: np.isclose(
                df["gridded_emissions"], df["results"], atol=0.0, rtol=0.0001
            ),
        )
        emi_check_df.to_csv(
            v3_inv_v3_grid_path / f"{group_name}_v3_emi_check.csv", index=False
        )
        if not all(emi_check_df["qc_pass"]):
            print("QC FAILED:")
            # break
            # display(emi_check_df[emi_check_df["qc_pass"] == False])
            print(f"\tare all emis eq: {emi_check_df["emi_eq"].all()}")
            print(f"\tare all grid eq: {emi_check_df["grid_eq"].all()}")
            continue
        else:
            print(f"QC PASSED")

        annual_summed_da = xr.DataArray(
            np.flip(annual_summed_arr, axis=1),
            dims=["time", "y", "x"],
            coords={
                "time": years,
                "y": gepa_profile.y,
                "x": gepa_profile.x,
            },
            name=group_name,
            attrs={
                "units": "kt",
                "long_name": f"CH4 emissions from {group_name} gridded to 1km x 1km",
                "source": group_name,
                "description": f"CH4 emissions from {group_name} gridded to 1km x 1km",
            },
        )

        fig, ax1 = plt.subplots()
        color = "xkcd:lavender"
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Inventory Emissions (kt)", color=color)
        ax1.plot(emi_check_df["year"], emi_check_df["ghgi_ch4_kt_x"], color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax2 = ax1.twinx()
        color = "xkcd:orange"
        ax2.set_ylabel("Relative Difference (%)", color=color)
        ax2.plot(emi_check_df["year"], emi_check_df["rel_diff"], color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        fig.suptitle(
            f"Comparison of Inventory Emissions and\nGridded Emissions for {group_name}"
        )
        fig.tight_layout()
        plt.savefig(
            v3_inv_v3_grid_path / f"{group_name}_v3_emi_check.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        write_tif_output(annual_summed_da, tif_kt_output_path)

        # STEP 5.2: QC FLUX AGAINST V2 ----------------------------------------------
        ch4_flux_result_da = calculate_flux(annual_summed_da)
        flux_emi_qc_df = QC_flux_emis(ch4_flux_result_da, group_name, v2_name=v2_name)
        plot_annual_raster_data(ch4_flux_result_da, group_name)
        plot_raster_data_difference(ch4_flux_result_da, group_name)
        # STEP 6: SAVE THE FILES ----------------------------------------------------
        write_tif_output(annual_summed_da, tif_kt_output_path)
        IPCC_ID, SOURCE_NAME = group_name.split("_", maxsplit=1)

        netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
        netcdf_description = (
            f"Gridded EPA Inventory - {group_name} - "
            f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
        )
        # netcdf_title = f"CH4 emissions from {source_name} gridded to 1km x 1km"
        # write_ncdf_output(
        #     ch4_flux_result_da,
        #     nc_flux_output_path,
        #     netcdf_title,
        #     netcdf_description,
        # )
        # %%

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
