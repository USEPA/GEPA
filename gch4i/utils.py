import calendar
import concurrent
import logging
import threading
import time
import warnings
from pathlib import Path

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
from geocube.api.core import make_geocube
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.plot import show
from rasterio.profiles import default_gtiff_profile
from rasterio.warp import reproject
from tqdm.auto import tqdm

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


# define a function to normalize the population data by state and year
def normalize(x):
    return x / x.sum() if x.sum() > 0 else 0


# NOTE: xarray does not like the if else of the default normalize method, so I have
# defined a new one here to appease xarray.
def normalize_xr(x):
    return x / x.sum()


def check_raster_proxy_time_geo(emi_df, proxy_ds, row, geo_col, time_col):
    # check that the proxy and emi files have matching state years
    # NOTE: this is going to arise when the proxy data are lacking adequate
    # spatial or temporal coverage. Failure here will require finding new data
    # and/or filling in missing state/years.

    if geo_col is not None:
        grouper_cols = [geo_col, time_col]
    else:
        grouper_cols = [time_col]

    if time_col == "year_month":
        proxy_ds = proxy_ds.assign_coords(
            year_month=pd.to_datetime(
                pd.DataFrame(
                    {
                        "year": proxy_ds.year.values,
                        "month": proxy_ds.month.values,
                        "day": 1,
                    }
                )
            ).dt.strftime("%Y-%m")
        )

    rel_emi_check = (
        proxy_ds["rel_emi"]
        .groupby(grouper_cols)
        .sum()
        .to_dataframe()
        .reset_index()
        # .astype({time_col: int})
        .set_index(grouper_cols)
    )
    final_df = emi_df.set_index(grouper_cols).join(rel_emi_check)
    match_check = final_df[(final_df.ghgi_ch4_kt > 0) & (final_df.rel_emi < 0.9)]
    if len(match_check) > 0:

        if geo_col:
            missing_states = match_check.state_code.value_counts()
            logging.critical(
                f"QC FAILED: {row.emi_id}, {row.proxy_id}\n"
                "proxy state/year columns do not match emissions\n"
                "missing states and counts of missing times:\n"
                f"{missing_states.to_string().replace("\n", "\n\t")}"
                "\n"
            )
        else:
            logging.critical(
                f"QC FAILED: {row.emi_id}, {row.proxy_id}\n"
                "proxy time column does not match emissions\n"
                "missing times:\n"
                f"{match_check[match_check['has_proxy'] == False][time_col].unique()}"
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
    p_df: gpd.GeoDataFrame, timestep: str = "year"
) -> dict[str, np.array]:
    """grids allocation of emissions for a proxy"""

    gepa_profile = GEPA_spatial_profile()

    # if the proxy data do not nest nicely into rasters cells, we have to do some
    # disaggregation of the vectors to get the sums to equal during rasterization
    # so we need the cell geodataframe to do that. Only load the cell_gdf if we need
    # to do disaggregation. project to equal are for are calculations
    # if not (proxy_gdf.geometry.type == "Points").all():
    cell_gdf = get_cell_gdf().to_crs("ESRI:102003")

    # create a dictionary to hold the rasterized emissions
    ch4_kt_result_rasters = {}
    for time_var, time_data in p_df.groupby(timestep):
        print(time_var)

        # orig_emi_val = data["allocated_ch4_kt"].sum()
        # if we need to disaggregate non-point data. Maybe extra, but this accounts for
        # potential edge cases where the data change from year to year.

        if not (time_data.geometry.type == "Points").all():
            data_to_concat = []
            for geom_type, geom_data in time_data.groupby(time_data.geom_type):
                print(geom_type)
                # regular points can just be passed on "as is"
                if geom_type == "Point":
                    # get the point data
                    print("doing point data")
                    point_data = geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                    print(point_data.is_empty.any())
                    data_to_concat.append(point_data)
                # if we have multipoint data, we need to disaggregate the emissions
                # across the individual points and align them with the cells
                elif geom_type == "MultiPoint":
                    print("doing multipoint data")
                    # ref_val = geom_data["allocated_ch4_kt"].sum()
                    multi_point_data = (
                        geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                        .to_crs("ESRI:102003")
                        .explode(index_parts=True)
                        .reset_index()
                        .assign(
                            allocated_ch4_kt=lambda df: df.groupby("level_0")[
                                "allocated_ch4_kt"
                            ].transform(lambda x: x / len(x)),
                        )
                        .overlay(cell_gdf)
                        .reset_index()
                        .assign(
                            geometry=lambda df: df.centroid,
                        )
                        .groupby("index")
                        .agg({"allocated_ch4_kt": "sum", "geometry": "first"})
                    )
                    multi_point_data = gpd.GeoDataFrame(
                        multi_point_data, geometry="geometry", crs="ESRI:102003"
                    ).to_crs(4326)
                    # print(ref_val)
                    # print(multi_point_data["allocated_ch4_kt"].sum())
                    print(f" any empty data: {multi_point_data.is_empty.any()}")
                    data_to_concat.append(multi_point_data)
                # if the data are any kind of lines, compute the relative length within
                # each cell
                elif "Line" in geom_type:
                    print("doing line data")
                    line_data = (
                        geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                        # project to equal area for area calculations
                        .to_crs("ESRI:102003")
                        # calculate the original proxy length
                        .assign(orig_len=lambda df: df.length)
                        # overlay the proxy with the cells, this results in splitting the
                        # original proxies across any intersecting cells
                        .overlay(cell_gdf)
                        # # calculate the now partial proxy length, then divide the partial
                        # # proxy by the original proxy and multiply by the original
                        # # allocated emissions to the get the partial/disaggregated new emis.
                        .assign(
                            len=lambda df: df.length,
                            allocated_ch4_kt=lambda df: (df["len"] / df["orig_len"])
                            * df["allocated_ch4_kt"],
                            #     # make this new shape a point, that fits nicely inside the cell
                            #     # and we don't have to worry about boundary/edge issues with
                            #     # polygon / cell alignment
                            geometry=lambda df: df.centroid,
                        )
                        # back to 4326 for rasterization
                        .to_crs(4326).loc[:, ["geometry", "allocated_ch4_kt"]]
                    )
                    print(f" any empty data: {line_data.is_empty.any()}")
                    data_to_concat.append(line_data)
                # if the data are any kind of  polygons, compute the relative area in
                # each cell
                elif "Polygon" in geom_type:
                    print("doing polygon data")
                    polygon_data = (
                        geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                        # project to equal area for area calculations
                        .to_crs("ESRI:102003")
                        # calculate the original proxy area
                        .assign(orig_area=lambda df: df.area)
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
                    # print(polygon_data["allocated_ch4_kt"].sum())
                    print(f" any empty data: {polygon_data.is_empty.any()}")
                    data_to_concat.append(polygon_data)
                else:
                    raise ValueError("I don't support that geometry type. Need to fix")
            # concat the data back together
            ready_data = pd.concat(data_to_concat)
        else:
            ready_data = time_data.loc[:, ["geometry", "allocated_ch4_kt"]]

        # Check for empty or invalid geometries
        invalid_geometries = ready_data[
            ready_data.geometry.is_empty | ~ready_data.geometry.is_valid
        ]

        if not invalid_geometries.empty:
            logging.warning(
                f"Found {invalid_geometries.shape[0]} invalid or empty geometries with a total allocated_ch4_kt of "
                f"{invalid_geometries['allocated_ch4_kt'].sum()}"
            )
            # Drop invalid or empty geometries
            ready_data = ready_data[
                ~(ready_data.geometry.is_empty | ~ready_data.geometry.is_valid)
            ]

        print("compare pre and post processing sums:")
        print("pre data:  ", time_data[["allocated_ch4_kt"]].sum())
        print("post data: ", ready_data[["allocated_ch4_kt"]].sum())

        # now rasterize the emissions and sum within cells
        ch4_kt_raster = rasterize(
            shapes=[
                (shape, value)
                for shape, value in ready_data[["geometry", "allocated_ch4_kt"]].values
            ],
            out_shape=gepa_profile.arr_shape,
            fill=0,
            transform=gepa_profile.profile["transform"],
            dtype=np.float64,
            merge_alg=rasterio.enums.MergeAlg.add,
        )
        ch4_kt_result_rasters[time_var] = ch4_kt_raster

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


def calculate_flux(
    in_ds: xr.DataArray, timestep: str = "year", direction: str = "mass2flux"
) -> xr.DataArray:
    """calculates flux for dictionary of total emissions year/array pairs"""
    area_matrix = load_area_matrix()
    gepa_profile = GEPA_spatial_profile()
    times = in_ds.time.values

    def get_days_in_year(year):
        year = int(year)
        return 366 if calendar.isleap(year) else 365

    def get_days_in_month(year_month):
        year, month = year_month.split("-")
        year = int(year)
        month = int(month)
        return calendar.monthrange(year, month)[1]

    if timestep == "year_month":
        days_in_months = [get_days_in_month(x) for x in times]
        conv_factors = [calc_conversion_factor(x, area_matrix) for x in days_in_months]

        conv_ds = xr.DataArray(
            np.flip(conv_factors, 1),
            dims=["time", "y", "x"],
            coords=[times, gepa_profile.y, gepa_profile.x],
            name="conversion_factor",
        )
    elif timestep == "year":
        days_in_year = [get_days_in_year(x) for x in times]
        conv_factors = [calc_conversion_factor(x, area_matrix) for x in days_in_year]

        conv_ds = xr.DataArray(
            np.flip(conv_factors, 1),
            dims=["time", "y", "x"],
            coords=[times, gepa_profile.y, gepa_profile.x],
            name="conversion_factor",
        )
    else:
        raise ValueError(
            f"timestep must be either 'year' or 'year_month', not {timestep}"
        )

    if direction == "mass2flux":
        flux_da = in_ds * conv_ds
    elif direction == "flux2mass":
        flux_da = in_ds / conv_ds
    else:
        raise ValueError(
            f"direction must be either 'mass2flux' or 'flux2mass', not {direction}"
        )

    return flux_da


def QC_proxy_allocation(
    proxy_df, emi_df, row, geo_col, time_col, plot=True, plot_path=None
) -> pd.DataFrame:
    """take proxy emi allocations and check against state inventory"""
    # logging.info("checking proxy emission allocation by state / year.")

    if geo_col is not None:
        grouper_cols = [geo_col, time_col]
    else:
        grouper_cols = [time_col]

    emi_sums = emi_df.groupby(grouper_cols)["ghgi_ch4_kt"].sum().reset_index()

    relative_tolerance = 0.0001
    sum_check = (
        proxy_df.groupby(grouper_cols)["allocated_ch4_kt"]
        .sum()
        .reset_index()
        .merge(emi_sums, on=grouper_cols, how="outer")
        .assign(
            isclose=lambda df: np.isclose(df["allocated_ch4_kt"], df["ghgi_ch4_kt"]),
            diff=lambda df: np.abs(df["allocated_ch4_kt"] - df["ghgi_ch4_kt"])
            / ((df["allocated_ch4_kt"] + df["ghgi_ch4_kt"]) / 2),
            qc_pass=lambda df: (df["diff"] < relative_tolerance),
        )
    )

    # in the cases where the difference is NaN (where the inventory emission value is 0
    # and the allocated value is 0), we fall back on numpy isclose to define
    # if the value is close enough to pass
    sum_check["qc_pass"] = sum_check["isclose"].where(
        sum_check["diff"].isna(), sum_check["qc_pass"]
    )

    all_pass = sum_check.qc_pass.all()

    if all_pass:
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

    if plot and all_pass and geo_col and time_col:
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
    elif plot and all_pass and time_col and not geo_col:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        fig.suptitle("compare inventory to allocated emissions by state")
        sns.lineplot(
            data=sum_check,
            x=time_col,
            y="allocated_ch4_kt",
            legend=False,
            ax=axs[0],
        )
        axs[0].set(title="allocated emissions")

        sns.lineplot(
            data=sum_check,
            x=time_col,
            y="ghgi_ch4_kt",
            legend=False,
            ax=axs[1],
        )
        axs[1].set(title="inventory emissions")
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return all_pass, sum_check


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

    # The relative tolerance is the maximum allowable difference between the two values
    # as a fraction of the average of the two values. It is used to determine if
    # the two values are close enough to be considered equal.
    # The default value is 1e-5, which means that the two values must be within 0.01%
    # of each other to be considered equal.
    relative_tolerance = 0.0001
    sum_check = emi_sum_check.join(proxy_sum_check).assign(
        isclose=lambda df: np.isclose(df["ghgi_ch4_kt"], df["results"], rtol=1e-5),
        rel_diff=lambda df: np.abs(df["ghgi_ch4_kt"] - df["results"])
        / ((df["ghgi_ch4_kt"] + df["results"]) / 2),
        qc_pass=lambda df: (df["rel_diff"] < relative_tolerance),
    )

    # in the cases where the difference is NaN (where the inventory emission value is 0
    # and the allocated value is 0), we fall back on numpy isclose to define
    # if the value is close enough to pass
    sum_check["qc_pass"] = sum_check["isclose"].where(
        sum_check["rel_diff"].isna(), sum_check["qc_pass"]
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


def calc_conversion_factor(year_days: int, area_matrix: np.array) -> np.array:
    """calculate emissions in kt to flux"""
    return 10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix


def write_tif_output(in_ds: xr.Dataset, dst_path: Path, resolution=0.1) -> None:
    """Write a raster to a tif file from a xarray object

    Args:
        in_ds (xr.Dataset): Input xarray dataset containing the raster data.
        dst_path (Path): Path to the output tif file.
        resolution (float, optional): Resolution of the output raster. Defaults to 0.1.

    Returns:
        None

    NOTE: this is being used because the rioxarray.to_raster() function is broken. It
    itroduces noise into the array that I can't explain.

    """

    out_profile = GEPA_spatial_profile(resolution).profile

    out_data = np.flip(in_ds.values, axis=1)
    out_profile.update(count=out_data.shape[0])
    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(out_data)


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

    gepa_profile = GEPA_spatial_profile(resolution)
    min_year = np.min(in_da.time.values)
    max_year = np.max(in_da.time.values)

    data_xr = (
        in_da.rename({"y": "lat", "x": "lon"})
        .assign_coords({"lat": gepa_profile.y, "lon": gepa_profile.x})
        .rio.set_attrs(
            {
                "title": title,
                "description": description,
                "year": f"{min_year}-{max_year}",
                "units": units,
            }
        )
        .rio.write_crs(gepa_profile.profile["crs"])
        .rio.write_transform(gepa_profile.profile["transform"])
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
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
    # Plot the raster data for each year in the dictionary of rasters
    for year in v3_data.time:
        year = int(year.values)
        # subset the dict of rasters for each year
        raster_data = np.flip(v3_data.sel(time=year).values, 0)

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

        # Save the plots as PNG files to the figures directory
        plt.savefig(figures_data_dir_path / f"{SOURCE_NAME}_ch4_flux_{year}.png")

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
    plt.savefig(figures_data_dir_path / f"{SOURCE_NAME}_ch4_flux_difference.png")

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


def QC_flux_emis(v3_flux_da, SOURCE_NAME, v2_name=None) -> None:
    """
    Function to compare and plot the difference between v2 and v3 for each year of the
    raster data for each sector.
    """

    # Check for expected years
    from config import min_year, max_year

    gepa_profile = GEPA_spatial_profile()

    actual_years = set(v3_flux_da.time.values.astype(int))
    expected_years = set(range(min_year, max_year + 1))
    missing_years = expected_years - actual_years
    if missing_years:
        Warning(f"Missing years in v3 data for {SOURCE_NAME}: {sorted(missing_years)}")

    # Check for negative values
    for year in v3_flux_da.time:
        year_val = int(year.values)
        v3_arr = np.flip(v3_flux_da.sel(time=year).values, 0)
        neg_count = np.sum(v3_arr < 0)
        if neg_count > 0:
            neg_percent = (neg_count / v3_arr.size) * 100
            Warning(
                f"Source {SOURCE_NAME}, Year {year_val} has {neg_count} negative values ({neg_percent:.2f}% of cells)"
            )

    # compare against v2 values, if v2_data exists
    if v2_name == "":
        Warning(f"there is no v2 raster data to compare against v3 for {SOURCE_NAME}!")
    else:
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

        v2_flux_da = xr.DataArray(
            np.flip(np.array(list(v2_data_dict.values())), 1),
            dims=["time", "y", "x"],
            coords=[list(v2_data_dict.keys()), gepa_profile.y, gepa_profile.x],
            name=v2_name,
        )

        # %%
        v3_time_match_da = v3_flux_da.sel(time=list(v2_data_dict.keys()))
        flux_diff_da = v3_time_match_da - v2_flux_da

        v2_flux_yearly_sums = np.nansum(v2_flux_da.values, axis=(1, 2))
        v3_flux_yearly_sums = np.nansum(v3_time_match_da.values, axis=(1, 2))

        flux_dif_df = pd.DataFrame(
            {
                "year": list(v2_data_dict.keys()),
                "v2_sum": v2_flux_yearly_sums,
                "v3_sum": v3_flux_yearly_sums,
            }
        ).assign(metric="flux")
        flux_dif_df

        # %%
        v3_mass_da = calculate_flux(
            v3_time_match_da, timestep="year", direction="flux2mass"
        )
        v2_mass_da = calculate_flux(v2_flux_da, timestep="year", direction="flux2mass")
        mass_diff_da = v3_mass_da - v2_mass_da

        v2_mass_yearly_sums = np.nansum(v2_mass_da.values, axis=(1, 2))
        v3_mass_yearly_sums = np.nansum(v3_mass_da.values, axis=(1, 2))

        mass_dif_df = pd.DataFrame(
            {
                "year": list(v2_data_dict.keys()),
                "v2_sum": v2_mass_yearly_sums,
                "v3_sum": v3_mass_yearly_sums,
            }
        ).assign(metric="mass")
        mass_dif_df

        qc_df = pd.concat([flux_dif_df, mass_dif_df], axis=0).assign(
            yearly_dif=lambda df: df["v3_sum"] - df["v2_sum"],
            percent_dif=lambda df: 100
            * (df["v3_sum"] - df["v2_sum"])
            / ((df["v3_sum"] + df["v2_sum"]) / 2),
        )
        qc_df

        g = sns.relplot(
            kind="line",
            data=qc_df,
            x="year",
            y="percent_dif",
            hue="metric",
            palette="Set2",
        )
        g.figure.suptitle(f"{SOURCE_NAME} v3 vs v2 percent difference")
        g.set_axis_labels("Year", "Percent difference")
        # Save the figure
        g.savefig(
            figures_data_dir_path / f"{SOURCE_NAME}_v3_vs_v2_percent_difference.png"
        )
        plt.close()

        # for year in v3_flux_da.time:
        for year in flux_diff_da.time:
            year_val = int(year.values)
            yearly_dif = np.flip(flux_diff_da.sel(time=year).values, 1)

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
                transform=gepa_profile.profile["transform"],
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
                f"{year_val} Difference between v2 and v3 methane "
                f"emissions from {SOURCE_NAME}"
            )
            plt.title(difference_plot_title, fontsize=14)
            # Save the plot as a PNG file
            plt.savefig(
                figures_data_dir_path
                / f"{SOURCE_NAME}_ch4_flux_difference_v2_to_v3_{year_val}.png"
            )
            # Show the plot for review
            # plt.show()
            # Close the plot
            plt.close()

            # Plot the grid cell level frequencies of v2 and v3 methane emissions
            # for each year
            fig, (ax1, ax2) = plt.subplots(2)
            fig.tight_layout()
            ax1.hist(v2_data_dict[year_val].ravel(), bins=100)
            ax2.hist(v3_arr.ravel(), bins=100)
            # Add a title
            histogram_plot_title = (
                f"{year_val} Frequency of methane emissions from "
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
                + f"/{SOURCE_NAME}_ch4_flux_histogram_{year_val}.png"
            )
            # Show the plot for review
            # plt.show()
            # Close the plot
            plt.close()

        return qc_df


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
    lat_top = 55  # deg
    lat_bottom = 20  # deg
    # lon_left = -129.95  # deg
    # lon_right = -59.95  # deg
    # lat_top = 54.95  # deg
    # lat_bottom = 20.05  # deg
    valid_resolutions = [0.1, 0.01]

    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self.check_resolution(self.resolution)
        self.x = np.arange(self.lon_left, self.lon_right, self.resolution)
        self.y = np.arange(self.lat_top, self.lat_bottom, -self.resolution)
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
                resolution, 0.0, self.lon_left, 0.0, -resolution, self.lat_top
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
    resolution: float = 0.1,
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
    # xr_ds =xr.open_dataarray(input_path)
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
        .apply(normalize_xr)
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

    def emi_rasterize(e_df, adm_gdf, out_shape, transform, year):
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
            merge_alg=rasterio.enums.MergeAlg.add,
        )
        return year, emi_raster

    parallel = Parallel(n_jobs=-1, return_as="generator")
    emi_gen = parallel(
        delayed(emi_rasterize)(emi_time_df, admin_gdf, out_shape, transform, year)
        for year, emi_time_df in emi_df.groupby(time_col)
    )

    emi_rasters = []
    emi_years = []
    for year, raster in tqdm(
        emi_gen,
        total=time_total,
        desc="making emi grids",
    ):
        emi_rasters.append(raster)
        emi_years.append(year)

    emi_rasters_3d = np.stack(emi_rasters, axis=0)
    # flip the y axis as xarray expects
    emi_rasters_3d = np.flip(emi_rasters_3d, axis=1)

    emi_rasters = xr.DataArray(
        emi_rasters_3d,
        dims=[time_col, "y", "x"],
        coords={time_col: emi_years, "y": gepa_profile.y, "x": gepa_profile.x},
    )

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

    return emi_rasters


def fill_missing_year_months(proxy_ds, time_col):
    # Ensure proxy_ds has all year_months in the range 2012-2022
    # in cases where we have expanded the annual emissions to match a monthly proxy,
    # we need to ensure that if the proxy is missing any months, we assume there are
    # no emissions, but we still need a layer representing 0 in that month.
    # this will fill in any missign years
    if time_col == "year":
        proxy_year_month = proxy_ds["year"].values
        all_year_months = years
    elif time_col == "year_month":
        all_year_months = pd.date_range(
            start=f"{min(years)}-01", end=f"{max(years)}-12-31", freq="ME"
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
        if time_col == "year":
            for year in missing_year_months:
                if isinstance(proxy_ds.indexes["year_month"], pd.MultiIndex):
                    fill_value = (year, 1)
                else:
                    fill_value = year
                missing_da = xr.Dataset(
                    {"results": (["y", "x"], empty_array)},
                    coords={
                        "year_month": [fill_value],
                        "year": year,
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
            proxy_ds = proxy_ds.sortby("year")
        elif time_col == "year_month":
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
