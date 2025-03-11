import calendar
import concurrent
import threading
from pathlib import Path
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import osgeo  # noqa f401
import pandas as pd
import rasterio
import requests
import rioxarray  # noqa f401
import seaborn as sns
import xarray as xr
import time
from geocube.api.core import make_geocube
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.plot import show
from rasterio.profiles import default_gtiff_profile
from rasterio.warp import reproject
import logging
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from gch4i.config import (
    V3_DATA_PATH,
    figures_data_dir_path,
    global_data_dir_path,
    years,
)

logger = logging.getLogger(__name__)

# import warnings
Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
Molarch4 = 16.04  # CH4 molecular weight (g/mol)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes
# tg_scale = (
#    0.001  # Tg conversion factor
# )

# global warming potential of CH4 relative to CO2
# (used to convert mass to CO2e units, from IPPC AR4)
GWP_CH4 = 25
# EEM: add constants (note, we should try to do conversions using variable names, so
#      that we don't have constants hard coded into the scripts)

# TODO: a REVERSE FLUX CALCULATION FUNCTION
# This function will take the flux and convert it back to emissions (in kt) for a given
# area
# def reverse_flux_calculation():
#     pass


# TODO: write state / year inventory to GRID allocation, probably using geocube
# EEM: question - for sources where we go from the state down to the grid-level, will
# we still have this calculation step, or will we go straight to the rasterize step?


# define a function to normalize the population data by state and year
def normalize(x):
    return x / x.sum() if x.sum() > 0 else 0


# NOTE: xarray does not like the if else of the default normalize method, so I have
# defined a new one here to appease xarray.
def normalize_xr(x):
    return x / x.sum()


def check_state_year_match(emi_df, proxy_gdf, row, match_cols=["state_code", "year"]):

    # check if all the proxy state / time columns are in the emissions data
    # NOTE: this check happens again later, but at a siginificant time cost
    # if the data are large. It is here to catch the error early and break
    # the loop with critical message.
    proxy_unique = (
        proxy_gdf[match_cols]
        .drop_duplicates()
        .assign(has_proxy=True)
        .set_index(match_cols)
    )
    match_check = (
        emi_df[match_cols]
        .set_index(match_cols)
        .join(proxy_unique)
        .fillna({"has_proxy": False})
        .sort_values("has_proxy")
        .reset_index()
    )
    # if row.proxy_has_year_col & row.proxy_has_rel_emi_col:

    #     proxy_unique = proxy_gdf.groupby(match_cols)[row.proxy_rel_emi_col].sum()
    #     match_check = (
    #         emi_df.set_index(match_cols)
    #         .join(proxy_unique, how="left")
    #         .rename(columns={row.proxy_rel_emi_col: "proxy"})
    #     )
    # elif row.proxy_has_year_col & (not row.proxy_has_rel_emi_col):
    #     proxy_unique = proxy_gdf.groupby(match_cols)["geometry"].count().to_frame()
    #     match_check = (
    #         emi_df.set_index(match_cols)
    #         .join(proxy_unique, how="left")
    #         .rename(columns={"geometry": "proxy"})
    #     )
    # elif (not row.proxy_has_year_col) & (row.proxy_has_rel_emi_col):
    #     proxy_unique = proxy_gdf.groupby(["state_code"])[row.proxy_rel_emi_col].sum()
    #     match_check = (
    #         emi_df.groupby(["state_code"])["ghgi_ch4_kt"]
    #         .sum()
    #         .to_frame()
    #         .join(proxy_unique, how="left")
    #         .rename(columns={row.proxy_rel_emi_col: "proxy"})
    #     )
    # elif (not row.proxy_has_year_col) & (not row.proxy_has_rel_emi_col):

    #     proxy_unique = proxy_gdf.groupby(match_cols)["geometry"].count().to_frame()
    #     match_check = (
    #         emi_df.set_index(match_cols)
    #         .join(proxy_unique, how="left")
    #         .rename(columns={"geometry": "proxy"})
    #     )
    # else:
    #     raise ValueError("this should not happen")

    if not match_check.has_proxy.all():
        missing_states = (
            match_check[match_check["has_proxy"] == False].groupby("state_code").size()
        )
        logging.critical(
            f"QC FAILED: {row.emi_id}, {row.proxy_id}\n"
            "proxy state/year columns do not match emissions\n"
            "missing states and counts of missing times:\n"
            # f"{missing_states}\n"
            f"{missing_states.to_string().replace("\n", "\n\t")}"
            "\n"
        )

        return False, match_check
    else:
        return True, match_check


def allocate_emissions_to_proxy(
    proxy_gdf: gpd.GeoDataFrame,
    emi_df: pd.DataFrame,
    # proxy_has_year: bool = False,
    use_proportional: bool = True,
    proportional_col_name: str = None,
    match_cols: list[str] = ["state_code", "year"],
    # geo_col: str = "state_code",
    # time_col: str = "year",
) -> gpd.GeoDataFrame:
    """
    Allocation state emissions by year to all proxies within the state by year.
    NOTE: 2024-06-21: tested with ferro, composting, and aban coal

    Inputs:
        proxy_gdf:
            -   GeoDataFrame: vector proxy data with or without fractional emissions to
                be used in allocation from state inventory data
        emi_df:
            -   The EPA state level emissions per year, typically read in from
                the IndDB sheet in the excel workbook
        proxy_has_year:
            -   If the proxy data have a yearly proportional value to use
        use_proportional:
            -   Indicate if the proxy has fractional emissions to be used in the
                allocation of inventory emissions to the point. For each state /
                year, the fractional emissions of all points within the state /
                year are used to allocation inventory emissions.
        proportional_col_name:
            -   the name of the column with proportional emissions.
    Returns:
        -   GeoDataFrame with new column "allocated_ch4_kt" added to proxy_gdf

    """

    if use_proportional and (proportional_col_name is None):
        raise ValueError(
            "must provide 'proportional_col_name' if 'use_proportional' is True."
        )

    # if proxy_has_year and (time_col not in proxy_gdf.columns):
    #     raise ValueError(
    #         f"proxy data must have {time_col} column if 'proxy_has_year' is True."
    #     )

    # if geo_col not in proxy_gdf.columns:
    #     raise ValueError(f"proxy data must have {geo_col} column")

    # if geo_col not in emi_df.columns:
    #     raise ValueError(f"inventory data must have {geo_col} column")

    # if time_col not in emi_df.columns:
    #     raise ValueError(f"inventory data must have {time_col} column")

    # if time_col not in proxy_gdf.columns:
    #     raise ValueError(f"proxy data must have {time_col} column")

    out_proxy_gdf = (
        proxy_gdf[match_cols + ["geometry", proportional_col_name]]
        .merge(emi_df, on=match_cols, how="right")
        .assign(
            allocated_ch4_kt=lambda df: df[proportional_col_name] * df["ghgi_ch4_kt"]
        )
    )

    # result_list = []
    # # for each state and year in the inventory data
    # for (state, year), data in emi_df.groupby([geo_col, time_col]):
    #     # if the proxy has a year, get the proxies for that state / year
    #     if proxy_has_year:
    #         state_proxy_data = proxy_gdf[
    #             (proxy_gdf[geo_col] == state) & (proxy_gdf[time_col] == year)
    #         ].copy()
    #     # else just get the proxy in the state
    #     else:
    #         state_proxy_data = (
    #             proxy_gdf[(proxy_gdf[geo_col] == state)].copy().assign(year=year)
    #         )
    #     # if there are no proxies in that state, print a warning
    #     if state_proxy_data.shape[0] < 1:
    #         # logging.warning(
    #         #     f"there are no proxies in {state} for {year} but there are emissions!"
    #         # )
    #         continue
    #     # get the value of emissions for that state/year
    #     state_year_emissions = data["ghgi_ch4_kt"].iat[0]

    #     # if there are no state inventory emissions, assign all proxy for that
    #     # state / year as 0

    #     if state_year_emissions == 0:
    #         logging.info(
    #             f"there are proxies in {state} for {year} but there are no emissions!"
    #         )
    #         state_proxy_data["allocated_ch4_kt"] = 0
    #     # else, compute the emission for each proxy record
    #     else:
    #         # if the proxy has a proportional value, say from subpart data, use it
    #         if use_proportional:
    #             state_proxy_data["allocated_ch4_kt"] = (
    #                 (
    #                     state_proxy_data[proportional_col_name]
    #                     / state_proxy_data[proportional_col_name].sum()
    #                 )
    #                 * state_year_emissions
    #             ).fillna(0)
    #         # else allocate emissions equally to all proxies in state
    #         else:
    #             state_proxy_data["allocated_ch4_kt"] = (
    #                 state_year_emissions / state_proxy_data.shape[0]
    #             )
    #     result_list.append(state_proxy_data)
    # out_proxy_gdf = pd.concat(result_list).reset_index(drop=True)
    return out_proxy_gdf


def get_cell_gdf() -> gpd.GeoDataFrame:
    """create a geodata frame of cell polygons matching our output grid"""
    profile = GEPA_spatial_profile()
    # get the number of cells
    ncells = np.multiply(*profile.arr_shape)
    # create an empty array of the right shape, assign each cell a unique value
    tmp_arr = np.arange(ncells, dtype=np.int32).reshape(profile.arr_shape)

    # get the cells as individual items in a dictionary holding their id and geom
    results = [
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(
            shapes(tmp_arr, transform=profile.profile["transform"])
        )
    ]

    # turn geom dictionary into a geodataframe
    cell_gdf = gpd.GeoDataFrame.from_features(results, crs=4326).drop(
        columns="raster_val"
    )
    return cell_gdf


def grid_allocated_emissions(
    proxy_gdf: gpd.GeoDataFrame, timestep: str = "year"
) -> dict[str, np.array]:
    """grids allocation of emissions for a proxy"""

    gepa_profile = GEPA_spatial_profile()

    # if the proxy data do not nest nicely into rasters cells, we have to do some
    # disaggregation of the vectors to get the sums to equal during rasterization
    # so we need the cell geodataframe to do that. Only load the cell_gdf if we need
    # to do disaggregation. project to equal are for are calculations
    if not (proxy_gdf.geometry.type == "Points").all():
        cell_gdf = get_cell_gdf().to_crs("ESRI:102003")

    # create a dictionary to hold the rasterized emissions
    ch4_kt_result_rasters = {}
    for time_var, data in proxy_gdf.groupby(timestep):
        # orig_emi_val = data["allocated_ch4_kt"].sum()
        # if we need to disaggregate non-point data. Maybe extra, but this accounts for
        # potential edge cases where the data change from year to year.
        if not (data.geometry.type == "Points").all():
            data_to_concat = []

            # get the point data
            point_data = data.loc[
                (data.type == "Point"),
                ["geometry", "allocated_ch4_kt"],
            ]
            data_to_concat.append(point_data)

            # get the non-point data
            # if the data are lines, compute the relative length within each cell
            if data.type.str.contains("Line").any():
                line_data = (
                    (
                        data.loc[
                            data.type.str.contains("Line"),
                            ["geometry", "allocated_ch4_kt"],
                        ]
                        # project to equal area for area calculations
                        .to_crs("ESRI:102003")
                        # calculate the original proxy length
                        .assign(orig_len=lambda df: df.length)
                    )
                    # overlay the proxy with the cells, this results in splitting the
                    # original proxies across any intersecting cells
                    .overlay(cell_gdf)
                    # calculate the now partial proxy length, then divide the partial
                    # proxy by the original proxy and multiply by the original
                    # allocated emissions to the get the partial/disaggregated new emis.
                    .assign(
                        len=lambda df: df.length,
                        allocated_ch4_kt=lambda df: (df["len"] / df["orig_len"])
                        * df["allocated_ch4_kt"],
                        # make this new shape a point, that fits nicely inside the cell
                        # and we don't have to worry about boundary/edge issues with
                        # polygon / cell alignment
                        geometry=lambda df: df.centroid,
                    )
                    # back to 4326 for rasterization
                    .to_crs(4326).loc[:, ["geometry", "allocated_ch4_kt"]]
                )
                data_to_concat.append(line_data)

            # if the data are polygons, compute the relative area in each cell
            if data.type.str.contains("Polygon").any():
                polygon_data = (
                    (
                        data.loc[
                            data.type.str.contains("Polygon"),
                            ["geometry", "allocated_ch4_kt"],
                        ]
                        # project to equal area for area calculations
                        .to_crs("ESRI:102003")
                        # calculate the original proxy area
                        .assign(orig_area=lambda df: df.area)
                    )
                    # overlay the proxy with the cells, this results in splitting the
                    # original proxies across any intersecting cells
                    .overlay(cell_gdf)
                    # calculate the now partial proxy area, then divide the partial
                    # proxy by the original proxy and multiply by the original
                    # allocated emissions to the get the partial/disaggregated new emis.
                    .assign(
                        area=lambda df: df.area,
                        allocated_ch4_kt=lambda df: (df["area"] / df["orig_area"])
                        * df["allocated_ch4_kt"],
                        # make this new shape a point, that fits nicely inside the cell
                        # and we don't have to worry about boundary/edge issues with
                        # polygon / cell alignment
                        geometry=lambda df: df.centroid,
                    )
                    # back to 4326 for rasterization
                    .to_crs(4326).loc[:, ["geometry", "allocated_ch4_kt"]]
                )
                data_to_concat.append(polygon_data)
            # concat the data back together
            data = pd.concat(data_to_concat)

        # new_emi_val = data["allocated_ch4_kt"].sum()

        # now rasterize the emissions and sum within cells
        ch4_kt_raster = rasterize(
            shapes=[
                (shape, value)
                for shape, value in data[["geometry", "allocated_ch4_kt"]].values
            ],
            out_shape=gepa_profile.arr_shape,
            fill=0,
            transform=gepa_profile.profile["transform"],
            dtype=np.float64,
            merge_alg=rasterio.enums.MergeAlg.add,
        )
        ch4_kt_result_rasters[time_var] = ch4_kt_raster
        # QC print values during gridding.
        # print(f"orig emi val: {orig_emi_val}, new emi val: {new_emi_val}")
        # print(f"raster emi val: {ch4_kt_raster.sum()}")

    time_var = list(ch4_kt_result_rasters.keys())
    arr_stack = np.stack(list(ch4_kt_result_rasters.values()), axis=0)
    arr_stack = np.flip(arr_stack, axis=1)
    out_ds = xr.Dataset(
        {"results": ([timestep, "y", "x"], arr_stack)},
        coords={timestep: time_var, "y": gepa_profile.y, "x": gepa_profile.x},
    )
    if timestep == "year_month":
        out_ds = out_ds.assign_coords(
            year=(timestep, [int(x.split("-")[0]) for x in time_var]),
            month=(timestep, [int(x.split("-")[1]) for x in time_var]),
        )

    return out_ds


# TODO: write this to accept months. There will need to be an if/else statement
# to get the number of days used in the flux calculations.
# XXX: create an actual datetime column to use?
def calculate_flux(
    in_ds: dict[str, np.array], timestep: str = "year"
) -> dict[str, np.array]:
    """calculates flux for dictionary of total emissions year/array pairs"""
    area_matrix = load_area_matrix()

    gepa_profile = GEPA_spatial_profile()

    # for each year in the input dictionary
    ch4_flux_result_rasters = {}
    if timestep == "year":
        for time in in_ds.time:
            time_var = int(time)
            data = np.flip(in_ds.sel(time=time), 0)
            month_days = [calendar.monthrange(time_var, x)[1] for x in range(1, 13)]
            year_days = np.sum(month_days)
            conversion_factor_annual = calc_conversion_factor(year_days, area_matrix)
            ch4_flux_raster = data * conversion_factor_annual
            ch4_flux_result_rasters[time_var] = ch4_flux_raster
    else:
        raise ValueError("we can't do months yet...")

    time_var = list(ch4_flux_result_rasters.keys())
    arr_stack = np.stack(list(ch4_flux_result_rasters.values()), axis=0)
    arr_stack = np.flip(arr_stack, axis=1)
    out_ds = xr.DataArray(
        arr_stack,
        dims=["time", "y", "x"],
        coords=[time_var, gepa_profile.y, gepa_profile.x],
    ).rename("results")

    return out_ds


def QC_proxy_allocation(
    proxy_df, emi_df, row, match_cols, plot=True, plot_path=None
) -> pd.DataFrame:
    """take proxy emi allocations and check against state inventory"""
    # logging.info("checking proxy emission allocation by state / year.")

    if len(match_cols) == 1:
        geo_col = None
        time_col = "year"
    elif len(match_cols) == 2:
        geo_col = match_cols[0]
        time_col = match_cols[1]
    emi_sums = emi_df.groupby(match_cols)["ghgi_ch4_kt"].sum().reset_index()

    sum_check = (
        proxy_df.groupby(match_cols)["allocated_ch4_kt"]
        .sum()
        .reset_index()
        .merge(emi_sums, on=match_cols, how="outer")
        .assign(
            isclose=lambda df: df.apply(
                lambda x: np.isclose(x["allocated_ch4_kt"], x["ghgi_ch4_kt"]), axis=1
            )
        )
    )

    all_equal = sum_check["isclose"].all()
    if all_equal:
        logging.info("QC PASS: all proxy emission by state/year equal (isclose)")
    else:
        logging.critical(f"QC FAIL: {row.emi_id}, {row.proxy_id}. allocation failed.")
        logging.info("states and years with emissions that don't match")
        # logging.info(
        #     "\t" + sum_check[~sum_check["isclose"]].to_string().replace("\n", "\n\t")
        # )
        if geo_col:
            unique_state_codes = emi_df[
                ~emi_df["state_code"].isin(proxy_df["state_code"])
            ]["state_code"].unique()
            logging.warning(
                f"states with no proxy points in them: {unique_state_codes}"
            )
            logging.info(
                (
                    "states with unaccounted emissions: "
                    f"{sum_check[~sum_check['isclose']]['state_code'].unique()}"
                )
            )
        else:
            logging.critical(
                f"QC FAIL: {row.emi_id}, {row.proxy_id}. annual allocation failed."
            )

    if plot and all_equal and geo_col and time_col:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        fig.suptitle("compare inventory to allocated emissions by state")
        sns.lineplot(
            data=sum_check,
            x=time_col,
            y="allocated_ch4_kt",
            hue=geo_col,
            palette="tab20",
            legend=False,
            ax=axs[0],
        )
        axs[0].set(title="allocated emissions")

        sns.lineplot(
            data=sum_check,
            x=time_col,
            y="ghgi_ch4_kt",
            hue=geo_col,
            palette="tab20",
            legend=False,
            ax=axs[1],
        )
        axs[1].set(title="inventory emissions")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    elif plot and all_equal and time_col and not geo_col:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        fig.suptitle("compare inventory to allocated emissions by state")
        sns.lineplot(
            data=sum_check,
            x=time_col,
            y="allocated_ch4_kt",
            # hue=geo_col,
            palette="tab20",
            legend=False,
            ax=axs[0],
        )
        axs[0].set(title="allocated emissions")

        sns.lineplot(
            data=sum_check,
            x=time_col,
            y="ghgi_ch4_kt",
            # hue=geo_col,
            palette="tab20",
            legend=False,
            ax=axs[1],
        )
        axs[1].set(title="inventory emissions")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return all_equal, sum_check


def QC_emi_raster_sums(
    proxy_da: xr.DataArray, emi_df: pd.DataFrame, timestep: str
) -> pd.DataFrame:
    """compares yearly array sums to inventory emissions"""

    # logging.info("checking gridded emissions result by year.")
    proxy_sum_check = proxy_da.sum(dim=["x", "y"]).to_dataframe()

    if timestep == "year_month" and isinstance(proxy_sum_check.index, pd.MultiIndex):
        proxy_sum_check.index = proxy_sum_check.index.map(
            lambda x: f"{x[0]}-{x[1]:02d}"
        )

    emi_sum_check = emi_df.groupby(timestep)["ghgi_ch4_kt"].sum().to_frame()

    relative_tolerance = 0.0001
    sum_check = emi_sum_check.join(proxy_sum_check).assign(
        isclose=lambda df: np.isclose(df["ghgi_ch4_kt"], df["results"]),
        diff=lambda df: np.abs(df["ghgi_ch4_kt"] - df["ghgi_ch4_kt"])
        / ((df["ghgi_ch4_kt"] + df["ghgi_ch4_kt"]) / 2),
        qc_pass=lambda df: df["diff"] < relative_tolerance,
    )

    all_pass = sum_check.qc_pass.all()

    if all_pass:
        logging.info(f"QC PASS: all gridded emission by {timestep}.")
    else:
        logging.critical("QC FAIL: gridded emissions do not equal inventory emissions.")
        logging.info(
            "\t" + sum_check[~sum_check.qc_pass].to_string().replace("\n", "\n\t")
        )
    return all_pass, sum_check


# sum a stack of xarray dataarrays
def combine_gridded_emissions(da_list: list[xr.DataArray]) -> xr.DataArray:
    """
    Sum a list of xarray DataArrays along a specified dimension.

    Parameters:
    dataarrays (list[xr.DataArray]): List of xarray DataArrays to sum.

    Returns:
    xr.DataArray: Summed xarray DataArray.
    """
    if len(da_list) == 1:
        return da_list[0]
    else:
        return xr.concat(da_list, dim="time").sum(dim=["x", "y"])


# def combine_gridded_emissions(
#     input_list: list[dict[str, np.array]]
# ) -> dict[str, np.array]:
#     """takes a dictionary of year/array pair and sums the arrays by year

#     input:
#         -   list of dictionaries with key/value pairs of year and 2D arrays
#     output:
#         -   dictionary of year/2D array summation of emissions by year.

#     """
#     stack_list = []
#     for x in input_list:
#         stack = np.stack(list(x.values()))
#         stack_list.append(stack)
#     out_sum_stack = np.sum(np.stack(stack_list), axis=0)
#     out_dict = {}
#     for i, year in enumerate(input_list[0].keys()):
#         out_dict[year] = out_sum_stack[i, :, :]
#     return out_dict


def calc_conversion_factor(year_days: int, area_matrix: np.array) -> np.array:
    """calculate emissions in kt to flux"""
    return 10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix


def write_tif_output(in_ds: xr.Dataset, dst_path: Path, resolution=0.1) -> None:
    """take an input dictionary with year/array items, write raster to dst_path"""

    gepa_profile = GEPA_spatial_profile(resolution)

    in_ds.rio.write_crs(gepa_profile.profile["crs"]).rio.write_transform(
        gepa_profile.profile["transform"]
    ).rio.to_raster(dst_path)
    return None


def load_area_matrix(resolution=0.1) -> np.array:
    """load the raster array of grid cell area in square meters"""
    res_text = str(resolution).replace(".", "")
    input_path = global_data_dir_path / f"gridded_area_{res_text}_cm2.tif"
    with rasterio.open(input_path) as src:
        arr = src.read(1)
    return arr


def write_ncdf_output(
    in_da: xr.DataArray,
    dst_path: Path,
    description: str,
    title: str,
    units: str = "moleccm-2s-1",
    resolution: float = 0.1,
    # month_flag: bool = False,
) -> None:
    """take dict of year:array pairs and write to dst_path with attrs"""
    # year_list = list(raster_dict.keys())
    # array_stack = np.stack(list(raster_dict.values()))

    # TODO: update function for this
    # if month_flag:
    #     pass

    # TODO: accept different resolutions for this function
    # if resolution:
    #     pass

    profile = GEPA_spatial_profile(resolution)
    min_year = np.min(in_da.time.values)
    max_year = np.max(in_da.time.values)

    data_xr = (
        in_da.rename({"y": "lat", "x": "lon"})
        .rio.set_attrs(
            {
                "title": title,
                "description": description,
                "year": f"{min_year}-{max_year}",
                "units": units,
            }
        )
        .rio.write_crs(profile.profile["crs"])
        .rio.write_transform(profile.profile["transform"])
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        # I think this only write out the year names in a separate table. Doesn't
        # seem useful.
        # .rio.write_coordinate_system()
        .rio.write_nodata(0.0, encoded=True)
    )
    data_xr.to_netcdf(dst_path.with_suffix(".nc"))
    return None


def name_formatter(col: pd.Series) -> pd.Series:
    """standard name formatted to allow for matching between datasets

    casefold
    replace any repeated spaces with just one
    remove any non-alphanumeric chars

    input:
        col = pandas Series
    returns = pandas series
    """
    return (
        col.str.strip()
        .str.casefold()
        .str.replace("\s+", " ", regex=True)  # noqa w605
        .replace("[^a-zA-Z0-9 -]", "", regex=True)
    )


def plot_annual_raster_data(v3_data, SOURCE_NAME) -> None:
    """
    Function to plot the raster data for each year in the dictionary of rasters that are
    output at the end of each sector script.
    """

    profile = GEPA_spatial_profile()

    # Plot the raster data for each year in the dictionary of rasters
    for year in v3_data.time:
        year = int(year.values)
        # subset the dict of rasters for each year
        raster_data = np.flip(v3_data.sel(time=year).values, 0)

        # The EPA color map from their V2 plots
        custom_colormap = colors.LinearSegmentedColormap.from_list(
            name="custom_colormap",
            colors=[
                "#6F4C9B",
                "#6059A9",
                "#5568B8",
                "#4E79C5",
                "#4D8AC6",
                "#4E96BC",
                "#549EB3",
                "#59A5A9",
                "#60AB9E",
                "#69B190",
                "#77B77D",
                "#8CBC68",
                "#A6BE54",
                "#BEBC48",
                "#D1B541",
                "#DDAA3C",
                "#E49C39",
                "#E78C35",
                "#E67932",
                "#E4632D",
                "#DF4828",
                "#DA2222",
                "#B8221E",
                "#95211B",
                "#721E17",
                "#521A13",
            ],
            N=3000,
        )

        # Set all raster values == 0 to nan so they are not plotted
        raster_data[np.where(raster_data == 0)] = np.nan

        # Plot the raster with Cartopy
        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # Set extent to the continental US
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())

        # This is a "background map" workaround that allows us to add plot features like
        # the colorbar and then use rasterio to plot the raster data on top of the
        # background map.
        background_map = ax.imshow(
            raster_data,
            cmap=custom_colormap,
        )

        # Add natural earth features
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES)

        # Plot the raster data using rasterio (this uses matplotlib imshow under the
        # hood)
        show(
            raster_data,
            transform=profile.profile["transform"],
            ax=ax,
            cmap=custom_colormap,
            interpolation="none",
        )

        # Set various plot parameters
        ax.tick_params(labelsize=10)
        fig.colorbar(
            background_map,
            orientation="horizontal",
            label="Methane emissions (Mg a$^{-1}$ km$^{-2}$)",
        )

        # Add a title
        annual_plot_title = f"{year} EPA methane emissions from {SOURCE_NAME}"
        annual_plot_title = (
            f"{year} EPA methane emissions from {SOURCE_NAME.split('_')[-1]}"
        )
        plt.title(annual_plot_title, fontsize=14)

        # Save the plot as a PNG file
        # plt.savefig(figures_data_dir_path / f"{SOURCE_NAME}_ch4_flux_{year}.png")

        # Save the plots as PNG files to the figures directory
        plt.savefig(str(figures_data_dir_path) + f"/{SOURCE_NAME}_ch4_flux_{year}.png")

        # Show the plot for review
        # plt.show()

        # close the plot
        plt.close()


def plot_raster_data_difference(in_da: xr.DataArray, SOURCE_NAME) -> None:
    """
    Function to plot the difference between the first and last years of the raster data
    for each sector.
    """
    # Define the geographic transformation parameters

    profile = GEPA_spatial_profile()

    # Get the first and last years of the data
    list_of_data_years = list(in_da.time.values)
    first_year_data = in_da.sel(time=np.min(in_da.time.values))
    last_year_data = in_da.sel(time=np.max(in_da.time.values))
    first_year_data = np.flip(first_year_data.values, 0)
    last_year_data = np.flip(last_year_data.values, 0)

    # Calculate the difference between the first and last years
    difference_raster = last_year_data - first_year_data

    # Set all raster values == 0 to nan so they are not plotted
    difference_raster[np.where(difference_raster == 0)] = np.nan

    custom_colormap = colors.LinearSegmentedColormap.from_list(
        name="custom_colormap",
        colors=[
            "#2166AC",
            "#4393C3",
            "#92C5DE",
            "#D1E5F0",
            "#F7F7F7",
            "#FDDBC7",
            "#F4A582",
            "#D6604D",
            "#B2182B",
        ],
        N=3000,
    )

    # Create a figure and axis with the specified projection
    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Set extent to the continental US
    ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())

    # This is a "background map" workaround that allows us to add plot features like
    # the colorbar and then use rasterio to plot the raster data on top of the
    # background map.
    background_map = ax.imshow(
        difference_raster,
        cmap=custom_colormap,
    )

    # Add natural earth features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)

    # Plot the raster data using rasterio (this uses matplotlib imshow under the hood)
    show(
        difference_raster,
        transform=profile.profile["transform"],
        ax=ax,
        cmap=custom_colormap,
        interpolation="none",
    )

    # Set various plot parameters
    ax.tick_params(labelsize=10)
    fig.colorbar(
        background_map,
        orientation="horizontal",
        label="Methane emissions (Mg a$^{-1}$ km$^{-2}$)",
    )

    # Add a title
    difference_plot_title = (
        f"Difference between {list_of_data_years[0]} and "
        f"{list_of_data_years[-1]} methane emissions from {SOURCE_NAME}"
    )
    plt.title(difference_plot_title, fontsize=14)

    # Save the plot as a PNG file
    plt.savefig(str(figures_data_dir_path) + f"/{SOURCE_NAME}_ch4_flux_difference.png")

    # Show the plot for review
    # plt.show()

    # close the plot
    plt.close()


us_state_to_abbrev_dict = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}


def us_state_to_abbrev(state_name: str) -> str:
    """converts a full US state name to the two-letter abbreviation"""
    return (
        us_state_to_abbrev_dict[state_name]
        if state_name in us_state_to_abbrev_dict
        else state_name
    )


def QC_flux_emis(v3_data, SOURCE_NAME, v2_name=None) -> None:
    """
    Function to compare and plot the difference between v2 and v3 for each year of the
    raster data for each sector.
    """

    # Check for expected years
    from config import (min_year, max_year)
    actual_years = set(v3_data.time.values.astype(int))
    expected_years = set(range(min_year, max_year + 1))
    missing_years = expected_years - actual_years
    if missing_years:
        Warning(f"Missing years in v3 data for {SOURCE_NAME}: {sorted(missing_years)}")
    
    # Check for negative values
    for year in v3_data.time:
        year_val = int(year.values)
        v3_arr = np.flip(v3_data.sel(time=year).values, 0)
        neg_count = np.sum(v3_arr < 0)
        if neg_count > 0:
            neg_percent = (neg_count / v3_arr.size) * 100
            Warning(f"Source {SOURCE_NAME}, Year {year_val} has {neg_count} negative values ({neg_percent:.2f}% of cells)")

    # compare against v2 values, if v2_data exists
    if v2_name is None:
        Warning(f"there is no v2 raster data to compare against v3 for {SOURCE_NAME}!")
    else:
        profile = GEPA_spatial_profile()

        # Get v2 flux raster data
        v2_data_paths = V3_DATA_PATH.glob("Gridded_GHGI_Methane_v2_*.nc")
        v2_data_dict = {}
        # The v2 data are not projected, so we get warnings reading all these files
        # suppress the warnings about no georeference.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for in_path in v2_data_paths:
                v2_year = int(in_path.stem.split("_")[-1])
                v2_data = rioxarray.open_rasterio(in_path, variable=v2_name)[
                    v2_name
                ].values.squeeze(axis=0)
                v2_data_dict[v2_year] = v2_data

        # Compare v2 data against v3 data for available v2 years
        result_list = []
        # for year in v3_data.keys():
        for year in v3_data.time:
            year = int(year.values)
            if year in v2_data_dict.keys():
                v3_arr = np.flip(v3_data.sel(time=year).values, 0)
                # Comparison of fluxes:
                # difference flux raster
                yearly_dif = v3_arr - v2_data_dict[year]
                # v2 flux raster sum
                v2_sum = np.nansum(v2_data_dict[year])
                # v3 flux raster sum
                v3_sum = np.nansum(v3_arr)
                # percent difference between v2 and v3
                percent_dif = 100 * (v3_sum - v2_sum) / ((v3_sum + v2_sum) / 2)
                logging.info(
                    f"year: {year}, v2 flux sum: {v2_sum}, v3 flux sum: {v3_sum}, "
                    f"percent difference: {percent_dif}"
                )
                # descriptive statistics on the difference raster
                result_list.append(
                    pd.DataFrame(yearly_dif.ravel())
                    .dropna(how="all", axis=1)
                    .describe()
                    .rename(columns={0: year})
                )
                # Comparison of masses:
                # flux to mass conversion factor
                area_matrix = load_area_matrix()
                month_days = [
                    calendar.monthrange(int(year), x)[1] for x in range(1, 13)
                ]
                year_days = np.sum(month_days)
                conversion_factor_annual = calc_conversion_factor(
                    year_days, area_matrix
                )
                # v2 mass raster sum
                # divide by the flux conversion factor to transform back into mass units
                v2_mass_raster = v2_data_dict[year] / conversion_factor_annual
                v2_mass_sum = np.nansum(v2_mass_raster)
                # v3 mass raster sum
                v3_mass_raster = v3_arr / conversion_factor_annual
                v3_mass_sum = np.nansum(v3_mass_raster)
                # percent difference between v2 and v3
                percent_dif_mass = (
                    100
                    * (v3_mass_sum - v2_mass_sum)
                    / ((v3_mass_sum + v2_mass_sum) / 2)
                )
                logging.info(
                    f"year: {year}, v2 mass sum: {v2_mass_sum}, "
                    f"v3 mass sum: {v3_mass_sum}, "
                    f"percent difference: {percent_dif_mass}"
                )

                # Set all raster values == 0 to nan so they are not plotted
                v2_data_dict[year][np.where(v2_data_dict[year] == 0)] = np.nan
                v3_arr[np.where(v3_arr == 0)] = np.nan
                yearly_dif[np.where(yearly_dif == 0)] = np.nan

                # Plot the difference between v2 and v3 methane emissions for each year
                custom_colormap = colors.LinearSegmentedColormap.from_list(
                    name="custom_colormap",
                    colors=[
                        "#2166AC",
                        "#4393C3",
                        "#92C5DE",
                        "#D1E5F0",
                        "#F7F7F7",
                        "#FDDBC7",
                        "#F4A582",
                        "#D6604D",
                        "#B2182B",
                    ],
                    N=3000,
                )
                # Create a figure and axis with the specified projection
                fig, ax = plt.subplots(
                    figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
                )
                # Set extent to the continental US
                ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
                # This is a "background map" workaround that allows us to add plot
                # features like the colorbar and then use rasterio to plot the raster
                # data on top of the background map.
                background_map = ax.imshow(
                    yearly_dif,
                    cmap=custom_colormap,
                )
                # Add natural earth features
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.OCEAN)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.STATES)
                # Plot the raster data using rasterio (this uses matplotlib imshow
                # under the hood)
                show(
                    yearly_dif,
                    transform=profile.profile["transform"],
                    ax=ax,
                    cmap=custom_colormap,
                    interpolation="none",
                )
                # Set various plot parameters
                ax.tick_params(labelsize=10)
                fig.colorbar(
                    background_map,
                    orientation="horizontal",
                    label="Methane emissions (Mg a$^{-1}$ km$^{-2}$)",
                )
                # Add a title
                difference_plot_title = (
                    f"{year} Difference between v2 and v3 methane "
                    f"emissions from {SOURCE_NAME}"
                )
                plt.title(difference_plot_title, fontsize=14)
                # Save the plot as a PNG file
                plt.savefig(
                    str(figures_data_dir_path)
                    + f"/{SOURCE_NAME}_ch4_flux_difference_v2_to_v3_{year}.png"
                )
                # Show the plot for review
                # plt.show()
                # Close the plot
                plt.close()

                # Plot the grid cell level frequencies of v2 and v3 methane emissions
                # for each year
                fig, (ax1, ax2) = plt.subplots(2)
                fig.tight_layout()
                ax1.hist(v2_data_dict[year].ravel(), bins=100)
                ax2.hist(v3_arr.ravel(), bins=100)
                # Add a title
                histogram_plot_title = (
                    f"{year} Frequency of methane emissions from "
                    f"{SOURCE_NAME} at the grid cell level"
                )
                ax1.set_title(histogram_plot_title)
                # Add axis labels
                ax2.set_xlabel("Methane emissions (Mg a$^{-1}$ km$^{-2}$)")
                ax1.set_ylabel("v2 frequency")
                ax2.set_ylabel("v3 frequency")
                # Save the plot as a PNG file
                plt.savefig(
                    str(figures_data_dir_path)
                    + f"/{SOURCE_NAME}_ch4_flux_histogram_{year}.png"
                )
                # Show the plot for review
                # plt.show()
                # Close the plot
                plt.close()

        result_df = pd.concat(result_list, axis=1)
        return result_df


def download_url(url, output_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # print("File downloaded successfully!")
    except requests.exceptions.RequestException as e:
        # print("Error downloading the file:", e)
        raise e


def allocate_emis_to_array(proxy_ds, emi_df, row) -> np.array:

    GEPA_PROFILE = GEPA_spatial_profile()

    # this is a bandaid fix that has been noted for a future update.
    # When statefp is na, the grouper to allocated emissions skips that part of
    # the array, reducing the dimensions of the results and making the results
    # incompatible with other raster stacks. So we fill in the statefp with 0
    # for now.
    if row.emi_has_fips_col & row.emi_has_month_col:
        proxy_ds["geoid"] = proxy_ds["geoid"].fillna(0)
        group_cols = ["year", "geoid"]
        geo_col = "cntyfp"
    else:
        proxy_ds["statefp"] = proxy_ds["statefp"].fillna(0)
        group_cols = ["year", "statefp"]
        geo_col = "statefp"
    results = []
    for emi_row in emi_df.itertuples():

        try:
            arr = (
                proxy_ds.where(
                    (proxy_ds[geo_col] == emi_row.statefp)
                    & (proxy_ds["year"] == emi_row.year),
                    drop=True,
                )[row.proxy_rel_emi_col]
                * emi_row.ghgi_ch4_kt
            )
            results.append(arr)
        except Exception as e:
            print(e)
            continue

        # for _, emi_val in tmp_emi_df.iterrows():
        #     emi_val
        #     # print(month_emi)
        #     # print(month_emi["ghgi_ch4_kt"])
        #     # print(month_emi["month"])
        #     val = emi_val["ghgi_ch4_kt"]
        #     tmp = data[row.proxy_rel_emi_col] * val
        #     results.append(tmp)
        # tmp_emi_df
        # try:
        #     val = emi_df.query(f"{geo_col} == {geo_id} & year=={year}")[
        #         "ghgi_ch4_kt"
        #     ].values[0]
        #     # print(val)
        #     tmp = data[row.proxy_rel_emi_col] * val
        #     results.append(tmp)
        #     # print(f"res state {statefp}, {year}: {tmp.sum().values}, {val}")
        # except IndexError:
        #     # print(f"rel emi val = {data['rel_emi'].sum().values}")
        #     # if the state/year is missing from the emissions data, we assume 0
        #     # emissions
        #     tmp = data[row.proxy_rel_emi_col] * 0
        #     results.append(tmp)
        #     # print(f"missing {statefp}, {year}")
        #     # continue
    # put the results back together
    results_ds = (
        xr.concat(results, dim="stacked_year_y_x")
        .unstack("stacked_year_y_x")
        .rename("emissions")
        # .drop_vars("geoid")
        .sortby(["year", "y", "x"])
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        # .where(lambda x: x > 0)
        .rio.write_crs(4326)
        .rio.write_transform(proxy_ds.rio.transform())
        .rio.set_attrs(proxy_ds.attrs)
    )

    tmp_file = rasterio.MemoryFile(ext=".tif")
    results_ds.rio.to_raster(tmp_file.name, profile=GEPA_PROFILE.profile)
    results_ds = (
        rioxarray.open_rasterio(tmp_file)
        .rename({"band": "year"})
        .assign_coords(year=years)
        .fillna(0)
    )
    # results_ds.sel(year=2020).plot()
    # plt.show()

    def split_array(arr, keys):
        return {key: arr[i] for i, key in enumerate(keys)}

    arr_dict = split_array(results_ds.values, years)
    return arr_dict


class GEPA_spatial_profile:
    lon_left = -130  # deg
    lon_right = -60  # deg
    lat_up = 55  # deg
    lat_low = 20  # deg
    valid_resolutions = [0.1, 0.01]

    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self.check_resolution(self.resolution)
        self.x = np.arange(self.lon_left, self.lon_right, self.resolution)
        self.y = np.arange(self.lat_low, self.lat_up, self.resolution)
        self.height, self.width = self.arr_shape = (len(self.y), len(self.x))
        self.profile = self.get_profile(self.resolution)

    def check_resolution(self, resolution):
        if resolution not in self.valid_resolutions:
            raise ValueError(
                f"resolution must be one of {', '.join(self.valid_resolutions)}"
            )

    def get_profile(self, resolution):
        base_profile = default_gtiff_profile.copy()
        base_profile.update(
            transform=rasterio.Affine(
                resolution, 0.0, self.lon_left, 0.0, -resolution, self.lat_up
            ),
            height=self.height,
            width=self.width,
            crs=4326,
            dtype="float32",
        )
        return base_profile


def make_raster_binary(
    input_path: Path, output_path: Path, true_vals: np.array, num_workers: int = 1
):
    """take a raster file and convert it to a binary raster file based on the true_vals
    array"""
    with rasterio.open(input_path) as src:

        # Create a destination dataset based on source params. The
        # destination will be tiled, and we'll process the tiles
        # concurrently.
        profile = src.profile
        profile.update(nodata=-99, dtype="int8")

        with rasterio.open(output_path, "w", **profile) as dst:
            windows = [window for ij, window in dst.block_windows()]

            # We cannot write to the same file from multiple threads
            # without causing race conditions. To safely read/write
            # from multiple threads, we use a lock to protect the
            # DatasetReader/Writer
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(window):
                with read_lock:
                    src_array = src.read(window=window)

                # result = np.where((src_array >= 1) & (src_array <= 60), 1, 0)
                result = np.isin(src_array, true_vals)

                with write_lock:
                    dst.write(result, window=window)

            # We map the process() function over the list of
            # windows.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(process, windows)


def mask_raster_parallel(
    input_path: Path, output_path: Path, mask_path: Path, num_workers: int = 1
):
    """take a raster file and convert it to a binary raster file based on the true_vals
    array

    alpha version of the function. Not fully tested yet. Use with caution.

    This does not check if the mask raster is the same size as the input raster. It
    assumes that the mask raster is the same size as the input raster. If not, you will
    need to run the warp function with the input_path as the target_path to get the mask
    aligned correctly. Otherwise this will throw and error.
    """
    with rasterio.open(input_path) as src:
        with rasterio.open(mask_path) as msk:

            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            # profile = src.profile
            # profile.update(
            #     blockxsize=128, blockysize=128, tiled=True, nodata=0, dtype="float32"
            # )

            with rasterio.open(output_path, "w", **src.profile) as dst:
                windows = [window for ij, window in dst.block_windows()]

                # We cannot write to the same file from multiple threads
                # without causing race conditions. To safely read/write
                # from multiple threads, we use a lock to protect the
                # DatasetReader/Writer
                read_lock = threading.Lock()
                write_lock = threading.Lock()

                def process(window):
                    with read_lock:
                        src_array = src.read(window=window)
                        msk_array = msk.read(window=window)

                    # result = np.where((src_array >= 1) & (src_array <= 60), 1, 0)
                    result = np.where(msk_array == 1, src_array, 0)

                    with write_lock:
                        dst.write(result, window=window)

                # We map the process() function over the list of
                # windows.
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    executor.map(process, windows)


# take any input raster file and warp it to match the GEPA_PROFILE
def warp_to_gepa_grid(
    input_path: Path,
    output_path: Path,
    target_path: Path = None,
    resampling: str = "average",
    num_threads: int = 1,
):

    if target_path is None:
        # print("warping to GEPA grid")
        profile = GEPA_spatial_profile().profile
    else:
        # print("warping to other raster")
        with rasterio.open(target_path) as src:
            profile = src.profile

    try:
        resamp_method = getattr(Resampling, resampling)
    except AttributeError as ex:
        raise ValueError(
            f"resampling method {resampling} not found in rasterio.Resampling"
        ) from ex

    with rasterio.open(input_path) as src:
        profile.update(count=src.count, dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    dst_nodata=0.0,
                    resampling=resamp_method,
                    num_threads=num_threads,
                )


# take any vector data input and grid it to the standard GEPA grid
def vector_to_gepa_grid():
    pass


def stack_rasters(input_paths: list[Path], output_path: Path):
    profile = GEPA_spatial_profile().profile
    raster_list = []
    years = [int(x.name.split("_")[2]) for x in input_paths]
    for in_file in input_paths:
        if not in_file.exists():
            continue
        with rasterio.open(in_file) as src:
            raster_list.append(src.read(1))

    output_data = np.stack(raster_list, axis=0)

    profile.update(count=len(raster_list))

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_data)
        dst.descriptions = tuple([str(x) for x in years])


def proxy_from_stack(
    input_path: Path,
    state_geo_path: Path,
    output_path: Path,
):

    # read in the state file and filter to lower 48 + DC
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

    # read in the raw population data raster stack as a xarray dataset
    # masked will read the nodata value and set it to NA
    xr_ds = rioxarray.open_rasterio(input_path, masked=True).rename({"band": "year"})

    with rasterio.open(input_path) as src:
        ras_crs = src.crs

    # assign the band as our years so the output raster data has year band names
    xr_ds["year"] = years
    # remove NA values
    # pop_ds = pop_ds.where(pop_ds != -99999)

    # create a state grid to match the input population array
    # we use fill here to fill in the nodata values with 99 so that when we do the
    # groupby, the nodata area is not collapsed, and the resulting dimensions align
    # with the v3 gridded data.
    state_grid = make_geocube(
        vector_data=state_gdf, measurements=["statefp"], like=xr_ds, fill=99
    )
    state_grid
    # assign the state grid as a new variable in the population dataset
    xr_ds["statefp"] = state_grid["statefp"]
    xr_ds

    # plot the data to check
    xr_ds["statefp"].plot()

    # apply the normalization function to the population data
    out_ds = (
        xr_ds.groupby(["year", "statefp"])
        .apply(normalize)
        .sortby(["year", "y", "x"])
        .to_dataset(name="rel_emi")
    )
    out_ds["rel_emi"].shape

    # check that the normalization worked
    all_eq_df = (
        out_ds["rel_emi"]
        .groupby(["statefp", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .assign(
            # NOTE: Due to floating point rouding, we need to check if the sum is
            # close to 1, not exactly 1.
            is_close=lambda df: (np.isclose(df["sum_check"], 1))
            | (np.isclose(df["sum_check"], 0))
        )
    )

    vals_are_one = all_eq_df["is_close"].all()
    print(f"are all state/year norm sums equal to 1? {vals_are_one}")
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")

    # plot. Not hugely informative, but shows the data is there.
    out_ds["rel_emi"].sel(year=2020).plot.imshow()

    out_ds["rel_emi"].transpose("year", "y", "x").round(10).rio.write_crs(
        ras_crs
    ).to_netcdf(output_path)


def geocode_address(df, address_column):
    """
    Geocode addresses using Nominatim, handling missing values

    Parameters:
    df: DataFrame containing addresses
    address_column: Name of column containing addresses

    Returns:
    DataFrame with added latitude and longitude columns
    """
    # Check if address column exists
    if address_column not in df.columns:
        raise ValueError(f"Column {address_column} not found in DataFrame")

    # Initialize geocoder
    geolocator = Nominatim(user_agent="my_app")

    # Create cache dictionary
    geocode_cache = {}

    def get_lat_long(address):
        # Handle missing values
        if pd.isna(address) or str(address).strip() == "":
            return (None, None)

        # Check cache first
        if address in geocode_cache:
            return geocode_cache[address]

        try:
            # Add delay to respect Nominatim's usage policy
            time.sleep(1)
            location = geolocator.geocode(str(address))
            if location:
                result = (location.latitude, location.longitude)
                geocode_cache[address] = result
                return result
            return (None, None)

        except (GeocoderTimedOut, GeocoderServiceError):
            return (None, None)

    # Create lat_long column
    df["lat_long"] = None

    # Check if longitude column exists
    if "longitude" not in df.columns:
        df["longitude"] = None

    # Only geocode rows where both latitude and longitude are missing
    mask = (
        (df["longitude"].isna() | (df["longitude"] == ""))
        & (df["latitude"].isna() | (df["latitude"] == ""))
        & df[address_column].notna()
    )

    # Apply geocoding only to rows that need it
    df.loc[mask, "lat_long"] = df.loc[mask, address_column].apply(get_lat_long)

    # Update latitude/longitude only for rows that were geocoded
    df.loc[mask, "latitude"] = df.loc[mask, "lat_long"].apply(
        lambda x: x[0] if x else None
    )
    df.loc[mask, "longitude"] = df.loc[mask, "lat_long"].apply(
        lambda x: x[1] if x else None
    )

    # Drop temporary column
    df = df.drop("lat_long", axis=1)

    return df


def create_final_proxy_df(proxy_df):
    """
    Function to create the final proxy df that is ready for gridding

    Parameters:
    - proxy_df: DataFrame containing proxy data.

    Returns:
    - final_proxy_df: DataFrame containing the processed emissions data for each state.
    """

    # Create a GeoDataFrame and generate geometry from longitude and latitude
    proxy_gdf = gpd.GeoDataFrame(
        proxy_df,
        geometry=gpd.points_from_xy(
            proxy_df["longitude"], proxy_df["latitude"], crs="EPSG:4326"
        ),
    )

    # subset to only include the columns we want to keep
    proxy_gdf = proxy_gdf[["state_code", "year", "emis_kt", "geometry"]]

    # Normalize relative emissions to sum to 1 for each year and state
    proxy_gdf = proxy_gdf.groupby(["state_code", "year"]).filter(
        lambda x: x["emis_kt"].sum() > 0
    )  # drop state-years with 0 total volume
    proxy_gdf["rel_emi"] = proxy_gdf.groupby(["year", "state_code"])[
        "emis_kt"
    ].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )  # normalize to sum to 1
    sums = proxy_gdf.groupby(["state_code", "year"])[
        "rel_emi"
    ].sum()  # get sums to check normalization
    # assert that the sums are close to 1
    assert np.isclose(
        sums, 1.0, atol=1e-8
    ).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Drop the original emissions column
    proxy_gdf = proxy_gdf.drop(columns=["emis_kt"])

    # Rearrange columns
    proxy_gdf = proxy_gdf[["state_code", "year", "rel_emi", "geometry"]]

    return proxy_gdf


def scale_emi_to_month(proxy_gdf, emi_df, row):
    """
    Function to scale the emissions data to a monthly basis
    Parameters:

    - proxy_gdf: GeoDataFrame containing proxy data with relative emissions.
    - emi_df: DataFrame containing emissions data with monthly values.
    Returns:
    - month_check: DataFrame containing the check for monthly emissions.
    """
    # calculate the relative MONTHLY proxy emissions
    logging.info("Calculating monthly scaling factors for emissions data")

    monthly_scaling = (
        proxy_gdf.assign(
            year_month=lambda df: pd.to_datetime(
                df[["year", "month"]].assign(DAY=1)
            ).dt.strftime("%Y-%m"),
        )
        .groupby(["state_code", "year_month"])["annual_rel_emi"]
        .sum()
        .rename("month_scale")
        .reset_index()
        .assign(
            year=lambda df: df["year_month"].str.split("-").str[0],
            month_normed=lambda df: df.groupby(["state_code", "year"])[
                "month_scale"
            ].transform(normalize),
        )
        .drop(columns=["month_scale", "year"])
        .set_index(["state_code", "year_month"])
    )
    monthly_scaling

    tmp_df = (
        emi_df.sort_values(["state_code", "year"])
        .assign(month=lambda df: [list(range(1, 13)) for _ in range(df.shape[0])])
        .explode("month")
        .reset_index(drop=True)
        .assign(
            year_month=lambda df: pd.to_datetime(
                df[["year", "month"]].assign(DAY=1)
            ).dt.strftime("%Y-%m"),
        )
        .set_index(["state_code", "year_month"])
        .join(
            monthly_scaling,
            how="right",
        )
        .assign(
            ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * df["month_normed"],
        )
        .drop(columns=["month_normed"])
        .reset_index()
    )
    tmp_df

    month_check = (
        tmp_df.groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .rename("month_check")
        .to_frame()
        .join(emi_df.set_index(["state_code", "year"]))
        .assign(
            isclose=lambda df: df.apply(
                lambda x: np.isclose(x["month_check"], x["ghgi_ch4_kt"]), axis=1
            )
        )
    )
    month_check["isclose"].all()

    if not month_check["isclose"].all():
        logging.critical("Monthly emissions do not sum to the expected values")
        raise ValueError(
            "Monthly emissions do not sum to the expected values. Check the log for "
            "details."
        )
    else:
        logging.info("Monthly emissions check passed!")
    return tmp_df


def make_emi_grid(emi_df, admin_gdf, time_col):
    """
    Function to create a 3D array of emissions data for each year and month
    Parameters:
    - emi_df: DataFrame containing emissions data with yearly or monthly values.
    - admin_gdf: GeoDataFrame containing administrative boundaries.
    Returns:
    - emi_rasters_3d: 3D array of emissions data for each year or year + month.
    """
    # to calculate emissions for a proxy array, we make the emissions into an
    # array of the same shape as the proxy array, and then multiply the two
    gepa_profile = GEPA_spatial_profile()
    transform = gepa_profile.profile["transform"]
    out_shape = gepa_profile.arr_shape
    time_total = emi_df[time_col].unique().shape[0]
    # this turns each year of the emissions data into a raster of the same shape
    # as the proxy in x and y. We have already aligned the time steps so the
    # number of emissions rasters should match the number of proxy rasters in
    # that dimension.
    # NOTE: this is kind of slow.

    def emi_rasterize(e_df, adm_gdf, out_shape, transform):
        emi_gdf = adm_gdf.merge(e_df, on="fips")
        emi_raster = rasterize(
            [
                (geom, value)
                for geom, value in zip(emi_gdf.geometry, emi_gdf.ghgi_ch4_kt)
            ],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="float32",
        )
        return emi_raster

    parallel = Parallel(n_jobs=-1, return_as="generator")
    emi_gen = parallel(
        delayed(emi_rasterize)(emi_time_df, admin_gdf, out_shape, transform)
        for _, emi_time_df in emi_df.groupby(time_col)
    )

    emi_rasters = []
    for raster in tqdm(
        emi_gen,
        total=time_total,
        desc="making emi grids",
    ):
        emi_rasters.append(raster)

    # for timestep, emi_time_df in tqdm(
    #     emi_df.groupby(time_col),
    #     total=time_total,
    #     desc="making emi grids",
    # ):
    #     emi_gdf = admin_gdf.merge(emi_time_df, on="fips")
    #     emi_raster = rasterize(
    #         [
    #             (geom, value)
    #             for geom, value in zip(emi_gdf.geometry, emi_gdf.ghgi_ch4_kt)
    #         ],
    #         out_shape=out_shape,
    #         transform=transform,
    #         fill=0,
    #         dtype="float32",
    #     )
    #     # Append the rasterized data to the list
    #     emi_rasters.append(emi_raster)
    # Stack the emi_raster objects into a 3D array
    # put the time axis at the end as xarray expects
    emi_rasters_3d = np.stack(emi_rasters, axis=0)
    # flip the y axis as xarray expects
    emi_rasters_3d = np.flip(emi_rasters_3d, axis=1)
    return emi_rasters_3d


def fill_missing_year_months(proxy_ds):
    # Ensure proxy_ds has all year_months in the range 2012-2022
    # in cases where we have expanded the annual emissions to match a monthly proxy,
    # we need to ensure that if the proxy is missing any months, we assume there are
    # no emissions, but we still need a layer representing 0 in that month.
    # this will fill in any missign years
    all_year_months = pd.date_range(
        start=f"{min(years)}-01", end=f"{max(years)}-12", freq="ME"
    ).strftime("%Y-%m")

    if isinstance(proxy_ds.indexes["year_month"], pd.MultiIndex):
        proxy_year_month = proxy_ds.indexes["year_month"].map(
            lambda x: f"{x[0]}-{x[1]:02d}"
        )
    else:
        proxy_year_month = proxy_ds["year_month"].values

    missing_year_months = set(all_year_months) - set(proxy_year_month)

    if missing_year_months:
        logging.info(f"Filling missing year_months: {missing_year_months}")
        empty_array = np.zeros_like(proxy_ds["results"].isel(year_month=0).values)
        for year_month in missing_year_months:
            year, month = map(int, year_month.split("-"))
            if isinstance(proxy_ds.indexes["year_month"], pd.MultiIndex):
                fill_value = (year, month)
            else:
                fill_value = year_month
            
            missing_da = xr.Dataset(
                {"results": (["y", "x"], empty_array)},
                coords={
                    "year_month": [fill_value],
                    "year": year,
                    "month": month,
                    "y": proxy_ds["y"],
                    "x": proxy_ds["x"],
                },
            )
            proxy_ds = xr.concat(
                [
                    proxy_ds,
                    missing_da,
                ],
                dim="year_month",
            )
        proxy_ds = proxy_ds.sortby("year_month")
    return proxy_ds
