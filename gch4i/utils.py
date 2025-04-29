from pathlib import Path
import calendar

import osgeo  # noqa f401
import numpy as np
import rasterio
from rasterio.plot import show
import geopandas as gpd
import xarray as xr
import pandas as pd
import rioxarray  # noqa f401
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from IPython.display import display
from rasterio.features import rasterize, shapes
import seaborn as sns

import gc
from datetime import datetime

from pyproj import CRS
from shapely.geometry import box

from gch4i.gridding import GEPA_spatial_profile

# import warnings

from gch4i.config import global_data_dir_path, figures_data_dir_path, V3_DATA_PATH, max_year, min_year, load_state_ansi, load_road_globals
from gch4i.gridding import GEPA_spatial_profile
from gch4i.config import RoadProxyGlobals

Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
Molarch4 = 16.04  # CH4 molecular weight (g/mol)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes
# tg_scale = (
#    0.001  # Tg conversion factor
# )
GWP_CH4 = 25  # global warming potential of CH4 relative to CO2 (used to convert mass to CO2e units, from IPPC AR4)
GWP_CH4 = 25  # global warming potential of CH4 relative to CO2 (used to convert mass
# to CO2e units, from IPPC AR4)
# EEM: add constants (note, we should try to do conversions using variable names, so
#      that we don't have constants hard coded into the scripts)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes
t_to_kt = 1000  # conversion factor, tonnes to kilotonnes


def calc_conversion_factor(days_in_year: int, cell_area_matrix: np.array) -> float:
    return (
        10**9
        * Avogadro
        / float(Molarch4 * days_in_year * 24 * 60 * 60)
        / cell_area_matrix
    )


# TODO: a REVERSE FLUX CALCULATION FUNCTION
# def reverse_flux_calculation():               # This function will take the flux and convert it back to emissions (in kt) for a given area
#     pass

# TODO: write state / year inventory to GRID allocation, probably using geocube
# EEM: question - for sources where we go from the state down to the grid-level, will
# we still have this calculation step, or will we go straight to the rasterize step?


def benchmark_load(func):
    """decorator to benchmark the load time of a function"""
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {datetime.now() - start}")
        return result
    return wrapper

def allocate_emissions_to_proxy(
    proxy_gdf: gpd.GeoDataFrame,
    emi_df: pd.DataFrame,
    proxy_has_year: bool = False,
    use_proportional: bool = False,
    proportional_col_name: str = None,
    # proxy_has_month: bool = False, # TODO: update code to have month.
    # emi_column = "state_code",
    # date_col: str = "year",
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

    if proxy_has_year and ("year" not in proxy_gdf.columns):
        raise ValueError(
            "proxy data must have 'year' column if 'proxy_has_year' is True."
        )

    if "state_code" not in proxy_gdf.columns:
        raise ValueError("proxy data must have 'state_code' column")

    if "state_code" not in emi_df.columns:
        raise ValueError("inventory data must have 'state_code' column")

    if "year" not in emi_df.columns:
        raise ValueError("inventory data must have 'year' column")

    result_list = []
    # for each state and year in the inventory data
    for (state, year), data in emi_df.groupby(["state_code", "year"]):
        # if the proxy has a year, get the proxies for that state / year
        if proxy_has_year:
            state_proxy_data = proxy_gdf[
                (proxy_gdf["state_code"] == state) & (proxy_gdf["year"] == year)
            ].copy()
        # else just get the proxy in the state
        else:
            state_proxy_data = (
                proxy_gdf[(proxy_gdf["state_code"] == state)].copy().assign(year=year)
            )
        # if there are no proxies in that state, print a warning
        if state_proxy_data.shape[0] < 1:
            Warning(
                f"there are no proxies in {state} for {year} but there are emissions!"
            )
            continue
        # get the value of emissions for that state/year
        state_year_emissions = data["ghgi_ch4_kt"].iat[0]
        # if there are no state inventory emissions, assign all proxy for that
        # state / year as 0
        if state_year_emissions == 0:
            Warning(
                f"there are proxies in {state} for {year} but there are no emissions!"
            )
            state_proxy_data["allocated_ch4_kt"] = 0
        # else, compute the emission for each proxy record
        else:
            # if the proxy has a proportional value, say from subpart data, use it
            if use_proportional:
                state_proxy_data["allocated_ch4_kt"] = (
                    (
                        state_proxy_data[proportional_col_name]
                        / state_proxy_data[proportional_col_name].sum()
                    )
                    * state_year_emissions
                ).fillna(0)
            # else allocate emissions equally to all proxies in state
            else:
                state_proxy_data["allocated_ch4_kt"] = (
                    state_year_emissions / state_proxy_data.shape[0]
                ).fillna(0)
        result_list.append(state_proxy_data)
    out_proxy_gdf = pd.concat(result_list).reset_index(drop=True)
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

    profile = GEPA_spatial_profile()
    ch4_kt_result_rasters = {}

    # if the proxy data do not nest nicely into rasters cells, we have to do some
    # disaggregation of the vectors to get the sums to equal during rasterization
    # so we need the cell geodataframe to do that. Only load the cell_gdf if we need
    # to do disaggregation. project to equal are for are calculations
    if not (proxy_gdf.geometry.type == "Points").all():
        cell_gdf = get_cell_gdf().to_crs("ESRI:102003")

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
            out_shape=profile.arr_shape,
            fill=0,
            transform=profile.profile["transform"],
            dtype=np.float64,
            merge_alg=rasterio.enums.MergeAlg.add,
        )
        ch4_kt_result_rasters[time_var] = ch4_kt_raster
        # QC print values during gridding.
        # print(f"orig emi val: {orig_emi_val}, new emi val: {new_emi_val}")
        # print(f"raster emi val: {ch4_kt_raster.sum()}")

    return ch4_kt_result_rasters


# TODO: write this to accept months. There will need to be an if/else statement
# to get the number of days used in the flux calculations.
# XXX: create an actual datetime column to use?
def calculate_flux(
    raster_dict: dict[str, np.array], timestep: str = "year"
) -> dict[str, np.array]:
    """calculates flux for dictionary of total emissions year/array pairs"""
    area_matrix = load_area_matrix()
    ch4_flux_result_rasters = {}
    if timestep == "year":
        for time_var, data in raster_dict.items():
            month_days = [calendar.monthrange(time_var, x)[1] for x in range(1, 13)]
            year_days = np.sum(month_days)
            conversion_factor_annual = calc_conversion_factor(year_days, area_matrix)
            ch4_flux_raster = data * conversion_factor_annual
            ch4_flux_result_rasters[time_var] = ch4_flux_raster
    else:
        raise ValueError("we can't do months yet...")
    return ch4_flux_result_rasters


def QC_proxy_allocation(proxy_df, emi_df, plot=True) -> pd.DataFrame:
    """take proxy emi allocations and check against state inventory"""
    print("checking proxy emission allocation by state / year.")
    sum_check = (
        proxy_df.groupby(["state_code", "year"])["allocated_ch4_kt"]
        .sum()
        .reset_index()
        .merge(emi_df, on=["state_code", "year"], how="outer")
        .assign(
            isclose=lambda df: df.apply(
                lambda x: np.isclose(x["allocated_ch4_kt"], x["ghgi_ch4_kt"]), axis=1
            )
        )
    )

    all_equal = sum_check["isclose"].all()
    print(f"do all proxy emission by state/year equal (isclose): {all_equal}")
    if not all_equal:
        print("states and years with emissions that don't match")
        display(sum_check[~sum_check["isclose"]])

        unique_state_codes = emi_df[~emi_df["state_code"].isin(proxy_df["state_code"])][
            "state_code"
        ].unique()

        print(f"states with no proxy points in them: {unique_state_codes}")
        print(
            (
                "states with unaccounted emissions: "
                f"{sum_check[~sum_check['isclose']]['state_code'].unique()}"
            )
        )
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        fig.suptitle("compare inventory to allocated emissions by state")
        sns.lineplot(
            data=sum_check,
            x="year",
            y="allocated_ch4_kt",
            hue="state_code",
            palette="tab20",
            legend=False,
            ax=axs[0],
        )
        axs[0].set(title="allocated emissions")

        sns.lineplot(
            data=sum_check,
            x="year",
            y="ghgi_ch4_kt",
            hue="state_code",
            palette="tab20",
            legend=False,
            ax=axs[1],
        )
        axs[1].set(title="inventory emissions")

    return sum_check


def QC_emi_raster_sums(raster_dict: dict, emi_df: pd.DataFrame) -> pd.DataFrame:
    """compares yearly array sums to inventory emissions"""

    print("checking gridded emissions result by year.")
    check_sum_dict = {}
    for year, arr in raster_dict.items():
        gridded_sum = arr.sum()
        check_sum_dict[year] = gridded_sum

    gridded_year_sums_df = (
        pd.DataFrame()
        .from_dict(check_sum_dict, orient="index")
        .rename(columns={0: "gridded_sum"})
    )

    sum_check = (
        emi_df.groupby("year")["ghgi_ch4_kt"]
        .sum()
        .to_frame()
        .join(gridded_year_sums_df)
        .assign(isclose=lambda df: np.isclose(df["ghgi_ch4_kt"], df["gridded_sum"]))
    )
    all_equal = sum_check["isclose"].all()

    print(f"do all gridded emission by year equal (isclose): {all_equal}")
    if not all_equal:
        print("if not, these ones below DO NOT equal")
        display(sum_check[~sum_check["isclose"]])
    return sum_check


def combine_gridded_emissions(
    input_list: list[dict[str, np.array]]
) -> dict[str, np.array]:
    """takes a dictionary of year/array pair and sums the arrays by year

    input:
        -   list of dictionaries with key/value pairs of year and 2D arrays
    output:
        -   dictionary of year/2D array summation of emissions by year.

    """
    stack_list = []
    for x in input_list:
        stack = np.stack(list(x.values()))
        stack_list.append(stack)
    out_sum_stack = np.sum(np.stack(stack_list), axis=0)
    out_dict = {}
    for i, year in enumerate(input_list[0].keys()):
        out_dict[year] = out_sum_stack[i, :, :]
    return out_dict


def calc_conversion_factor(year_days: int, area_matrix: np.array) -> np.array:
    """calculate emissions in kt to flux"""
    return 10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix


def write_tif_output(in_dict: dict, dst_path: Path, resolution=0.1) -> None:
    """take an input dictionary with year/array items, write raster to dst_path"""
    out_array = np.stack(list(in_dict.values()))

    profile = GEPA_spatial_profile(resolution)

    dst_profile = profile.profile.copy()

    dst_profile.update(count=out_array.shape[0])
    with rasterio.open(dst_path.with_suffix(".tif"), "w", **dst_profile) as dst:
        dst.write(out_array)
        dst.descriptions = [str(x) for x in in_dict.keys()]
    return None


def load_area_matrix(resolution=0.1) -> np.array:
    """load the raster array of grid cell area in square meters"""
    res_text = str(resolution).replace(".", "")
    input_path = global_data_dir_path / f"gridded_area_{res_text}_cm2.tif"
    with rasterio.open(input_path) as src:
        arr = src.read(1)
    return arr


def write_ncdf_output(
    raster_dict: dict,
    dst_path: Path,
    description: str,
    title: str,
    units: str = "moleccm-2s-1",
    resolution: float = 0.1,
    # month_flag: bool = False,
) -> None:
    """take dict of year:array pairs and write to dst_path with attrs"""
    year_list = list(raster_dict.keys())
    array_stack = np.stack(list(raster_dict.values()))

    # TODO: update function for this
    # if month_flag:
    #     pass

    # TODO: accept different resolutions for this function
    # if resolution:
    #     pass

    profile = GEPA_spatial_profile(resolution)

    data_xr = (
        xr.DataArray(
            array_stack,
            coords={
                "time": year_list,
                "lat": profile.y,
                "lon": profile.x,
            },
            dims=[
                "time",
                "lat",
                "lon",
            ],
        )
        .rio.set_attrs(
            {
                "title": title,
                "description": description,
                "year": f"{min(year_list)}-{max(year_list)}",
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


def plot_annual_raster_data(ch4_flux_result_rasters, SOURCE_NAME) -> None:
    """
    Function to plot the raster data for each year in the dictionary of rasters that are
    output at the end of each sector script.
    """

    profile = GEPA_spatial_profile()

    # Plot the raster data for each year in the dictionary of rasters
    for year in ch4_flux_result_rasters.keys():

        # subset the dict of rasters for each year
        raster_data = ch4_flux_result_rasters[year]

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
        plt.show()

        # close the plot
        plt.close()


def plot_raster_data_difference(ch4_flux_result_rasters, SOURCE_NAME) -> None:
    """
    Function to plot the difference between the first and last years of the raster data
    for each sector.
    """
    # Define the geographic transformation parameters

    profile = GEPA_spatial_profile()

    # Get the first and last years of the data
    list_of_data_years = list(ch4_flux_result_rasters.keys())
    first_year_data = ch4_flux_result_rasters[list_of_data_years[0]]
    last_year_data = ch4_flux_result_rasters[list_of_data_years[-1]]

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
    plt.show()

    # close the plot
    plt.close()


def QC_flux_emis(v3_data, SOURCE_NAME, v2_name) -> None:
    """
    Function to compare and plot the difference between v2 and v3 for each year of the raster data
    for each sector.
    """
    if v2_name is None:
        Warning(
            f"there is no v2 raster data to compare against v3 for {SOURCE_NAME}!"
            )
    else:
        profile = GEPA_spatial_profile()

        # Get v2 flux raster data
        v2_data_paths = V3_DATA_PATH.glob("Gridded_GHGI_Methane_v2_*.nc")
        v2_data_dict = {}
        for in_path in v2_data_paths:
            v2_year = int(in_path.stem.split("_")[-1])
            v2_data = rioxarray.open_rasterio(in_path, variable=v2_name)[
                v2_name
            ].values.squeeze(axis=0)
            v2_data_dict[v2_year] = v2_data

        # Compare v2 data against v3 data for available v2 years
        result_list = []
        for year in v3_data.keys():
            if year in v2_data_dict.keys():
                # Comparison of fluxes:
                # difference flux raster
                yearly_dif = v3_data[year] - v2_data_dict[year]
                # v2 flux raster sum
                v2_sum = np.nansum(v2_data)
                # v3 flux raster sum
                v3_sum = np.nansum(v3_data[year])
                # percent difference between v2 and v3
                percent_dif = 100*(v3_sum - v2_sum)/((v3_sum + v2_sum)/2)
                print(f"year: {year}, v2 flux sum: {v2_sum}, v3 flux sum: {v3_sum}, percent difference: {percent_dif}")
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
                month_days = [calendar.monthrange(int(year), x)[1] for x in range(1, 13)]
                year_days = np.sum(month_days)
                conversion_factor_annual = calc_conversion_factor(year_days, area_matrix)
                # v2 mass raster sum
                # divide by the flux conversion factor to transform back into mass units
                v2_mass_raster = v2_data_dict[year] / conversion_factor_annual
                v2_mass_sum = np.nansum(v2_mass_raster)
                # v3 mass raster sum
                v3_mass_raster = v3_data[year] / conversion_factor_annual
                v3_mass_sum = np.nansum(v3_mass_raster)
                # percent difference between v2 and v3
                percent_dif_mass = 100*(v3_mass_sum - v2_mass_sum)/((v3_mass_sum + v2_mass_sum)/2)
                print(f"year: {year}, v2 mass sum: {v2_mass_sum}, v3 mass sum: {v3_mass_sum}, percent difference: {percent_dif_mass}")

                # Set all raster values == 0 to nan so they are not plotted
                v2_data_dict[year][np.where(v2_data_dict[year] == 0)] = np.nan
                v3_data[year][np.where(v3_data[year] == 0)] = np.nan
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
                # This is a "background map" workaround that allows us to add plot features like
                # the colorbar and then use rasterio to plot the raster data on top of the
                # background map.
                background_map = ax.imshow(
                    yearly_dif,
                    cmap=custom_colormap,
                )
                # Add natural earth features
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.OCEAN)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.STATES)
                # Plot the raster data using rasterio (this uses matplotlib imshow under the hood)
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
                    f"{year} Difference between v2 and v3 methane emissions from {SOURCE_NAME}"
                )
                plt.title(difference_plot_title, fontsize=14)
                # Save the plot as a PNG file
                plt.savefig(str(figures_data_dir_path) + f"/{SOURCE_NAME}_ch4_flux_difference_v2_to_v3_{year}.png")
                # Show the plot for review
                plt.show()
                # Close the plot
                plt.close()
                
                # Plot the grid cell level frequencies of v2 and v3 methane emissions for each year
                fig, (ax1, ax2) = plt.subplots(2)
                fig.tight_layout()
                ax1.hist(v2_data_dict[year].ravel(), bins=100)
                ax2.hist(v3_data[year].ravel(), bins=100)
                # Add a title
                histogram_plot_title = f"{year} Frequency of methane emissions from {SOURCE_NAME} at the grid cell level"
                ax1.set_title(histogram_plot_title)
                # Add axis labels
                ax2.set_xlabel("Methane emissions (Mg a$^{-1}$ km$^{-2}$)")
                ax1.set_ylabel("v2 frequency")
                ax2.set_ylabel("v3 frequency")
                # Save the plot as a PNG file
                plt.savefig(str(figures_data_dir_path) + f"/{SOURCE_NAME}_ch4_flux_histogram_{year}.png")
                # Show the plot for review
                plt.show()
                # Close the plot
                plt.close()

        result_df = pd.concat(result_list, axis=1)
        return result_df


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



# ROAD PROXY FUNCTIONS
########################################################################################

# %% STEP 0.2 Load Path Files
raw_path, raw_roads_path, road_file, raw_road_file, task_outputs_path, global_path, gdf_state_files, global_input_path, state_ansi_path, GEPA_Comb_Mob_path, State_vmt_file,  State_vdf_file, State_ANSI, name_dict, state_mapping, start_year, end_year, year_range, year_range_str, num_years = load_road_globals()

def get_overlay_dir(year, 
                    out_dir: Path=task_outputs_path / 'overlay_cell_state_region'):
    return out_dir / f'cell_state_region_{year}.parquet'

def get_overlay_gdf(year, crs="EPSG:4326"):
    crs_obj = CRS(crs)

    gdf = gpd.read_parquet(get_overlay_dir(year))

    if gdf.crs != crs_obj:
        print(f"Converting overlay to crs {crs_obj.to_epsg()}")
        gdf.to_crs(crs, inplace=True)

    return gdf

# Read in State Spatial Data
def get_states_gdf(crs="EPSG:4326"):
    """
    Read in State spatial data
    """
    crs_obj = CRS(crs)

    gdf_states = gpd.read_file(gdf_state_files)


    gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
        ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
        )]
    gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]

    if gdf_states.crs != crs_obj:
        print(f"Converting states to crs {crs_obj.to_epsg()}")
        gdf_states.to_crs(crs, inplace=True)

    return gdf_states

# Read in Region Spatial Data
def get_region_gdf(year, crs=4326):
    """
    Read in region spatial data
    """
    crs = CRS(crs)
    road_loc = (
        gpd.read_parquet(f"{road_file}{year}_us_uac.parquet", columns=['geometry'])
        .assign(year=year)
        .to_crs(crs)
        .assign(urban=1)
    )
    return road_loc

def benchmark_load(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {datetime.now() - start}")
        return result
    return wrapper

def read_reduce_data(year):
    """
    Read in All Roads data. Deduplicate and reduce data early.
    """
    # check if file already exists
    out_path = task_outputs_path / f"reduced_roads_{year}.parquet"
    if out_path.exists():
        print(f'Found reduced roads file at path {out_path}.')
        print()
        return out_path
    # Order road types
    road_type_order = ['Primary', 'Secondary', 'Other']

    df = (gpd.read_parquet(f"{raw_road_file}{year}_us_allroads.parquet",
                           columns=['MTFCC', 'geometry'])
          .assign(year=year))

    road_data = (
        df.to_crs("ESRI:102003")
        .assign(
            geometry=lambda df: df.normalize(),
            road_type=lambda df: pd.Categorical(
                np.select(
                    [
                        df['MTFCC'] == 'S1100',
                        df['MTFCC'] == 'S1200',
                        df['MTFCC'].isin(['S1400', 'S1630', 'S1640'])
                    ],
                    [
                        'Primary',
                        'Secondary',
                        'Other'
                    ],
                    default=None
                ),
                categories=road_type_order,  # Define the categories
                ordered=True  # Ensure the categories are ordered
            )
        )
    )
    # Sort
    road_data = road_data.sort_values('road_type').reset_index(drop=True)
    # Explode to make LineStrings
    road_data = road_data.explode(index_parts=True).reset_index(drop=True)
    # Remove duplicates of geometries
    road_data = road_data.drop_duplicates(subset='geometry', keep='first')

    # Separate out Road Types
    prim_year = road_data[road_data['road_type'] == 'Primary']
    sec_year = road_data[road_data['road_type'] == 'Secondary']
    oth_year = road_data[road_data['road_type'] == 'Other']

    buffer_distance = 3  # meters

    # Set buffers
    prim_buffer = prim_year.buffer(buffer_distance)
    prim_buffer = gpd.GeoDataFrame(geometry=prim_buffer, crs=road_data.crs)

    prisec_buffer = pd.concat([prim_year, sec_year], ignore_index=True)
    prisec_buffer = prisec_buffer.buffer(buffer_distance)
    prisec_buffer = gpd.GeoDataFrame(geometry=prisec_buffer, crs=road_data.crs)

    # Overlay
    sec_red = gpd.overlay(sec_year, prim_buffer, how='difference')
    other_red = gpd.overlay(oth_year, prisec_buffer, how='difference')

    # Combine
    road_data = pd.concat([prim_year, sec_red, other_red], ignore_index=True)

    road_data = road_data[['year', 'road_type', 'geometry']]

    # Write to parquet
    road_data.to_parquet()

    del road_data, prim_year, sec_year, oth_year, prim_buffer, prisec_buffer, sec_red, other_red

    gc.collect()

    return out_path


def get_roads_path(year, raw_roads_path: Path=V3_DATA_PATH / "global/raw_roads", raw=True):
    '''
    If raw is True, it returns the parquet files of the original roads data at at path:
        C:/Users/<USER>/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/global/raw_roads/tl_{year}_us_allroads.parquet
    If raw is False, it returns the parquet files of the reduced roads data (as generated in Andrew's original task_roads_proxy.py script `reduce_roads()`): 
        C:/Users/<USER>/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/global/raw_roads/task_outputs/reduced_roads_{year}.parquet
    
    '''
    if raw:
        return Path(raw_roads_path) / f"tl_{year}_us_allroads.parquet"
    else:
        return Path(raw_roads_path) / f"task_outputs/reduced_roads_{year}.parquet"


@benchmark_load
def read_roads(year, raw=True, crs='EPSG:4326'):
    crs_obj = CRS(crs)
    gdf = gpd.read_parquet(get_roads_path(year, raw=raw))
    if gdf.crs != crs_obj:
        print(f"Converting {year} roads to crs {crs_obj.to_epsg()}")
        return gdf.to_crs(crs)
    else:
        return gdf
    
def intersect_sindex(cell, roads):
    '''
    Based of fthis geoff boeing blog post:
    https://geoffboeing.com/2016/10/r-tree-spatial-index-python/
    '''
    # first, add rtree spatial index to roads

    
    spatial_index = roads.sindex
    possible_matches_index = list(spatial_index.intersection(cell.bounds))
    possible_matches = roads.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(cell)]
    return precise_matches

# @benchmark_load
def intersect_and_clip(roads, state_geom, state_abbrev=None, simplify=None, dask=False):
    '''
    The simplify parameer is used to simplify the geometry of the roads before clipping to the state boundary
    '''
    state_box = box(*state_geom.bounds)
    # Get the roads that intersect with the state bounding box
    state_roads = intersect_sindex(state_box, roads)
    # print(f"Found {len(state_roads)} roads in {state_abbrev}")
    if not state_roads.empty:
        if simplify:
            state_geom = state_geom.simplify(simplify)
        

        # clip roads to state boundary
        result = state_roads.clip(state_geom)
        return result
    return state_roads


# Read in VM2 Data
def get_vm2_arrays(num_years):
    # Initialize arrays
    Miles_road_primary = np.zeros([2, len(State_ANSI), num_years])
    Miles_road_secondary = np.zeros([2, len(State_ANSI), num_years])
    Miles_road_other = np.zeros([2, len(State_ANSI), num_years])
    total = np.zeros(num_years)
    total2 = np.zeros(num_years)

    headers = ['STATE', 'RURAL - INTERSTATE', 'RURAL - FREEWAYS', 'RURAL - PRINCIPAL',
               'RURAL - MINOR', 'RURAL - MAJOR COLLECTOR', 'RURAL - MINOR COLLECTOR',
               'RURAL - LOCAL', 'RURAL - TOTAL', 'URBAN - INTERSTATE',
               'URBAN - FREEWAYS', 'URBAN - PRINCIPAL', 'URBAN - MINOR',
               'URBAN - MAJOR COLLECTOR', 'URBAN - MINOR COLLECTOR', 'URBAN - LOCAL',
               'URBAN - TOTAL', 'TOTAL']

    for iyear in np.arange(num_years):
        VMT_road = pd.read_excel(State_vmt_file + year_range_str[iyear] + '.xls',
                                 sheet_name='A',
                                 skiprows=13,
                                 nrows=51)
        VMT_road.columns = headers

        VMT_road = (VMT_road
                    .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                    .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                                   "District of Columbia" ""))
                    )

        for idx in np.arange(len(VMT_road)):
            VMT_road.loc[idx, 'ANSI'] = name_dict[VMT_road.loc[idx, 'STATE'].strip()]
            istate = np.where(VMT_road.loc[idx, 'ANSI'] == State_ANSI['ansi'])
            Miles_road_primary[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - INTERSTATE']
            Miles_road_primary[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - INTERSTATE']
            Miles_road_secondary[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - FREEWAYS'] + \
                VMT_road.loc[idx, 'URBAN - PRINCIPAL'] + \
                VMT_road.loc[idx, 'URBAN - MINOR']
            Miles_road_secondary[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - FREEWAYS'] + \
                VMT_road.loc[idx, 'RURAL - PRINCIPAL'] + \
                VMT_road.loc[idx, 'RURAL - MINOR']
            Miles_road_other[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - MAJOR COLLECTOR'] + \
                VMT_road.loc[idx, 'URBAN - MINOR COLLECTOR'] + \
                VMT_road.loc[idx, 'URBAN - LOCAL']
            Miles_road_other[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - MAJOR COLLECTOR'] + \
                VMT_road.loc[idx, 'RURAL - MINOR COLLECTOR'] + \
                VMT_road.loc[idx, 'RURAL - LOCAL']
            total[iyear] += np.sum(Miles_road_primary[:, istate, iyear]) + \
                np.sum(Miles_road_secondary[:, istate, iyear]) + \
                np.sum(Miles_road_other[:, istate, iyear])
            total2[iyear] += VMT_road.loc[idx, 'TOTAL']

        abs_diff = abs(total[iyear] - total2[iyear])/((total[iyear]+total2[iyear])/2)

        if abs(abs_diff) < 0.0001:
            print('Year ' + year_range_str[iyear] + ': Difference < 0.01%: PASS')
            print(total[iyear])
            print(total2[iyear])
        else:
            print('Year ' + year_range_str[iyear] + ': Difference > 0.01%: FAIL, diff: ' + str(abs_diff))
            print(total[iyear])
            print(total2[iyear])

    return Miles_road_primary, Miles_road_secondary, Miles_road_other, total, total2


# Read in VM4 Data
def get_vm4_arrays(num_years):
    # Initialize arrays
    Per_vmt_mot = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_pas = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_lig = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_hea = np.zeros([2, 3, len(State_ANSI), num_years])
    total_R = np.zeros(num_years)
    total_U = np.zeros(num_years)
    total = np.zeros(num_years)
    total2_U = np.zeros(num_years)
    total2 = np.zeros(num_years)
    total2_R = np.zeros(num_years)

    for iyear in np.arange(0, num_years):
        if year_range[iyear] == 2012 or year_range[iyear] == 2016:
            continue  # deal with missing data at the end
        else:
            # Read in Rural Sheet
            names = pd.read_excel(State_vdf_file + year_range_str[iyear]+'.xls',
                                  sheet_name='A', skiprows=12, header=0, nrows=1)
            colnames = names.columns.values
            VMT_type_R = pd.read_excel(State_vdf_file + year_range_str[iyear]+'.xls',
                                       na_values=['-'], sheet_name='A', names=colnames,
                                       skiprows=13, nrows=51)

            VMT_type_R.rename(columns={'MOTOR-': 'INTERSTATE - MOTORCYCLES',
                                       'PASSENGER': 'INTERSTATE - PASSENGER CARS',
                                       'LIGHT': 'INTERSTATE - LIGHT TRUCKS',
                                       'Unnamed: 4': 'INTERSTATE - BUSES',
                                       'SINGLE-UNIT': 'INTERSTATE - SINGLE-UNIT TRUCKS',
                                       'COMBINATION': 'INTERSTATE - COMBINATION TRUCKS',
                                       'Unnamed: 7': 'INTERSTATE - TOTAL',
                                       'MOTOR-.1': 'ARTERIALS - MOTORCYCLES',
                                       'PASSENGER.1': 'ARTERIALS - PASSENGER CARS',
                                       'LIGHT.1': 'ARTERIALS - LIGHT TRUCKS',
                                       'Unnamed: 11': 'ARTERIALS - BUSES',
                                       'SINGLE-UNIT.1': 'ARTERIALS - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.1': 'ARTERIALS - COMBINATION TRUCKS',
                                       'Unnamed: 14': 'ARTERIALS - TOTAL',
                                       'MOTOR-.2': 'OTHER - MOTORCYCLES',
                                       'PASSENGER.2': 'OTHER - PASSENGER CARS',
                                       'LIGHT.2': 'OTHER - LIGHT TRUCKS',
                                       'Unnamed: 18': 'OTHER - BUSES',
                                       'SINGLE-UNIT.2': 'OTHER - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.2': 'OTHER - COMBINATION TRUCKS',
                                       'Unnamed: 21': 'OTHER - TOTAL'}, inplace=True)

            VMT_type_R = (
                VMT_type_R
                .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                               "District of Columbia"))
                .assign(ANSI=0)
                .fillna(0)
                )

            # Read in Urban Sheet
            names = pd.read_excel(State_vdf_file + year_range_str[iyear] + '.xls',
                                  sheet_name='B', skiprows=12, header=0, nrows=1)
            colnames = names.columns.values
            VMT_type_U = pd.read_excel(State_vdf_file + year_range_str[iyear] + '.xls',
                                       na_values=['-'], sheet_name='B', names=colnames,
                                       skiprows=13, nrows=51)

            VMT_type_U.rename(columns={'MOTOR-': 'INTERSTATE - MOTORCYCLES',
                                       'PASSENGER': 'INTERSTATE - PASSENGER CARS',
                                       'LIGHT': 'INTERSTATE - LIGHT TRUCKS',
                                       'Unnamed: 4': 'INTERSTATE - BUSES',
                                       'SINGLE-UNIT': 'INTERSTATE - SINGLE-UNIT TRUCKS',
                                       'COMBINATION': 'INTERSTATE - COMBINATION TRUCKS',
                                       'Unnamed: 7': 'INTERSTATE - TOTAL',
                                       'MOTOR-.1': 'ARTERIALS - MOTORCYCLES',
                                       'PASSENGER.1': 'ARTERIALS - PASSENGER CARS',
                                       'LIGHT.1': 'ARTERIALS - LIGHT TRUCKS',
                                       'Unnamed: 11': 'ARTERIALS - BUSES',
                                       'SINGLE-UNIT.1': 'ARTERIALS - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.1': 'ARTERIALS - COMBINATION TRUCKS',
                                       'Unnamed: 14': 'ARTERIALS - TOTAL',
                                       'MOTOR-.2': 'OTHER - MOTORCYCLES',
                                       'PASSENGER.2': 'OTHER - PASSENGER CARS',
                                       'LIGHT.2': 'OTHER - LIGHT TRUCKS',
                                       'Unnamed: 18': 'OTHER - BUSES',
                                       'SINGLE-UNIT.2': 'OTHER - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.2': 'OTHER - COMBINATION TRUCKS',
                                       'Unnamed: 21': 'OTHER - TOTAL'}, inplace=True)

            VMT_type_U = (
                VMT_type_U
                .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                               "District of Columbia"))
                .assign(ANSI=0)
                .fillna(0)
                )

            # Distribute to 4 output types: passenger, light, heavy, motorcycle
            for idx in np.arange(len(VMT_type_R)):
                VMT_type_R.loc[idx, 'ANSI'] = name_dict[VMT_type_R.loc[idx, 'STATE']
                                                        .strip()]
                istate_R = np.where(VMT_type_R.loc[idx, 'ANSI'] == State_ANSI['ansi'])
                Per_vmt_mot[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - MOTORCYCLES']
                Per_vmt_mot[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - MOTORCYCLES']
                Per_vmt_mot[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - MOTORCYCLES']
                Per_vmt_pas[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - PASSENGER CARS']
                Per_vmt_pas[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - PASSENGER CARS']
                Per_vmt_pas[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - PASSENGER CARS']
                Per_vmt_lig[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - LIGHT TRUCKS']
                Per_vmt_lig[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - LIGHT TRUCKS']
                Per_vmt_lig[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - LIGHT TRUCKS']
                Per_vmt_hea[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - BUSES'] + \
                    VMT_type_R.loc[idx, 'INTERSTATE - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'INTERSTATE - COMBINATION TRUCKS']
                Per_vmt_hea[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - BUSES'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - COMBINATION TRUCKS']
                Per_vmt_hea[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - BUSES'] + \
                    VMT_type_R.loc[idx, 'OTHER - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'OTHER - COMBINATION TRUCKS']
                total_R[iyear] += np.sum(Per_vmt_mot[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_pas[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_lig[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_hea[1, :, istate_R, iyear])
                total2_R[iyear] += VMT_type_R.loc[idx, 'INTERSTATE - TOTAL'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - TOTAL'] + \
                    VMT_type_R.loc[idx, 'OTHER - TOTAL']

            for idx in np.arange(len(VMT_type_U)):
                VMT_type_U.loc[idx, 'ANSI'] = name_dict[VMT_type_U.loc[idx, 'STATE']
                                                        .strip()]
                istate_U = np.where(VMT_type_U.loc[idx, 'ANSI'] == State_ANSI['ansi'])
                Per_vmt_mot[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - MOTORCYCLES']
                Per_vmt_mot[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - MOTORCYCLES']
                Per_vmt_mot[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - MOTORCYCLES']
                Per_vmt_pas[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - PASSENGER CARS']
                Per_vmt_pas[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - PASSENGER CARS']
                Per_vmt_pas[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - PASSENGER CARS']
                Per_vmt_lig[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - LIGHT TRUCKS']
                Per_vmt_lig[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - LIGHT TRUCKS']
                Per_vmt_lig[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - LIGHT TRUCKS']
                Per_vmt_hea[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - BUSES'] + \
                    VMT_type_U.loc[idx, 'INTERSTATE - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'INTERSTATE - COMBINATION TRUCKS']
                Per_vmt_hea[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - BUSES'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - COMBINATION TRUCKS']
                Per_vmt_hea[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - BUSES'] + \
                    VMT_type_U.loc[idx, 'OTHER - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'OTHER - COMBINATION TRUCKS']
                total_U[iyear] += np.sum(Per_vmt_mot[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_pas[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_lig[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_hea[0, :, istate_U, iyear])
                total2_U[iyear] += VMT_type_U.loc[idx, 'INTERSTATE - TOTAL'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - TOTAL'] + \
                    VMT_type_U.loc[idx, 'OTHER - TOTAL']

            # Check for differences
            total[iyear] = total_U[iyear] + total_R[iyear]
            total2[iyear] = total2_R[iyear] + total2_U[iyear]
            abs_diff1 = abs(total[iyear] - total2[iyear]) / ((total[iyear] + total2[iyear]) / 2)

            if abs(abs_diff1) < 0.0001:
                print('Year ' + year_range_str[iyear] + ': Urban Difference < 0.01%: PASS')
            else:
                print('Year ' + year_range_str[iyear] + ': Urban Difference > 0.01%: FAIL, diff: ' + str(abs_diff1))
                print(total[iyear])
                print(total2[iyear])

    # Correct Years (assign 2012 to 2013), assign 2016 as average of 2015 and 2017
    idx_2012 = (2012-start_year)
    idx_2016 = (2016-start_year)
    Per_vmt_mot[:, :, :, idx_2012] = Per_vmt_mot[:, :, :, idx_2012 + 1]
    Per_vmt_pas[:, :, :, idx_2012] = Per_vmt_pas[:, :, :, idx_2012 + 1]
    Per_vmt_lig[:, :, :, idx_2012] = Per_vmt_lig[:, :, :, idx_2012 + 1]
    Per_vmt_hea[:, :, :, idx_2012] = Per_vmt_hea[:, :, :, idx_2012 + 1]

    Per_vmt_mot[:, :, :, idx_2016] = 0.5 * (Per_vmt_mot[:, :, :, idx_2016 - 1] +
                                            Per_vmt_mot[:, :, :, idx_2016 + 1])
    Per_vmt_pas[:, :, :, idx_2016] = 0.5 * (Per_vmt_pas[:, :, :, idx_2016 - 1] +
                                            Per_vmt_pas[:, :, :, idx_2016 + 1])
    Per_vmt_lig[:, :, :, idx_2016] = 0.5 * (Per_vmt_lig[:, :, :, idx_2016 - 1] +
                                            Per_vmt_lig[:, :, :, idx_2016 + 1])
    Per_vmt_hea[:, :, :, idx_2016] = 0.5 * (Per_vmt_hea[:, :, :, idx_2016 - 1] +
                                            Per_vmt_hea[:, :, :, idx_2016 + 1])

    # Optional: Combine Per_vmt_mot and Per_vmt_pas as Per_vmt_pas
    # Consult with EPA to determine whether to keep change
    Per_vmt_pas = Per_vmt_pas + Per_vmt_mot

    # Multiply by 0.01 to convert to percentage
    Per_vmt_mot = Per_vmt_mot * 0.01
    Per_vmt_pas = Per_vmt_pas * 0.01
    Per_vmt_lig = Per_vmt_lig * 0.01
    Per_vmt_hea = Per_vmt_hea * 0.01

    # Keep Per_vmt_mot reported for now, but it is now accounted for in Per_vmt_pas
    return Per_vmt_mot, Per_vmt_pas, Per_vmt_lig, Per_vmt_hea


# Calculate State Level Proxies
def calculate_state_proxies(num_years,
                            Miles_road_primary,
                            Miles_road_secondary,
                            Miles_road_other,
                            Per_vmt_pas,
                            Per_vmt_lig,
                            Per_vmt_hea):
    """
    array dimensions:
    region(urban/rural), road type(primary, secondary, other), state, year

    Example: vmt_pas : Miles_road_(primary, secondary, other) * Per_vmt_pas
    """

    # Initialize vmt_arrays
    vmt_pas = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_lig = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_hea = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_tot = np.zeros([2, len(State_ANSI), num_years])

    # Caclulate absolute number of VMT by region, road type, vehicle type, state, year
    # e.g. vmt_pas = VMT for passenger vehicles with dimensions = region (urban/rural),
    # road type (primary, secondary, other), state, and year

    # vmt_tot = region x state, year
    # road mile variable dimensions (urban/rural, state, year)

    for iyear in np.arange(0, num_years):
        vmt_pas[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_pas[:, 0, :, iyear]
        vmt_pas[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_pas[:, 1, :, iyear]
        vmt_pas[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_pas[:, 2, :, iyear]

        vmt_lig[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_lig[:, 0, :, iyear]
        vmt_lig[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_lig[:, 1, :, iyear]
        vmt_lig[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_lig[:, 2, :, iyear]

        vmt_hea[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_hea[:, 0, :, iyear]
        vmt_hea[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_hea[:, 1, :, iyear]
        vmt_hea[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_hea[:, 2, :, iyear]

        vmt_tot[:, :, iyear] += Miles_road_primary[:, :, iyear] + \
            Miles_road_secondary[:, :, iyear] + \
            Miles_road_other[:, :, iyear]

    # Initialize denominators
    tot_pas = np.zeros([len(State_ANSI), num_years])
    tot_lig = np.zeros([len(State_ANSI), num_years])
    tot_hea = np.zeros([len(State_ANSI), num_years])

    # Calculate total VMT for state/year by vehicle
    for istate in np.arange(0, len(name_dict)):
        for iyear in np.arange(0, num_years):
            tot_pas[istate, iyear] = np.sum(vmt_pas[:, :, istate, iyear])
            tot_lig[istate, iyear] = np.sum(vmt_lig[:, :, istate, iyear])
            tot_hea[istate, iyear] = np.sum(vmt_hea[:, :, istate, iyear])

    # Calculate proxy values
    print(f"CALCULATING")
    pas_proxy = vmt_pas / tot_pas
    lig_proxy = vmt_lig / tot_lig
    hea_proxy = vmt_hea / tot_hea

    return pas_proxy, lig_proxy, hea_proxy, vmt_tot


# Unpack State Proxy Arrays
def unpack_state_proxy(state_proxy_array, proxy_name='Emission Allocation'):
    reshaped_state_proxy = state_proxy_array.reshape(-1)

    row_index = np.repeat(['urban', 'rural'], 3 * 57 * num_years)
    col1_index = np.tile(np.repeat(['Primary', 'Secondary', 'Other'], 57 * num_years), 2)
    col2_index = np.tile(np.repeat(np.arange(1, 58), num_years), 2 * 3)
    col3_index = np.tile(np.arange(min_year, max_year + 1), 2 * 3 * 57)

    df = pd.DataFrame({
        'Region': row_index,
        'Road Type': col1_index,
        'State': col2_index,
        'Year': col3_index,
        proxy_name: reshaped_state_proxy
    })

    df['State_abbr'] = df['State'].map(state_mapping)

    cols = df.columns.tolist()
    state_index = cols.index('State')
    cols.insert(state_index + 1, cols.pop(cols.index('State_abbr')))
    df = df[cols]

    return df


# Unpack State Total Proxy Arrays
def unpack_state_allroads_proxy(vmt_tot):
    reshaped_state_proxy = vmt_tot.reshape(-1)

    row_index = np.repeat(['urban', 'rural'], 57 * num_years)
    col1_index = np.tile(np.repeat(np.arange(1, 58), num_years), 2)
    col2_index = np.tile(np.arange(min_year, max_year + 1), 2 * 57)

    df = pd.DataFrame({
        'Region': row_index,
        'State': col1_index,
        'Year': col2_index,
        'Proxy': reshaped_state_proxy
    })

    df['State_abbr'] = df['State'].map(state_mapping)

    cols = df.columns.tolist()
    state_index = cols.index('State')
    cols.insert(state_index + 1, cols.pop(cols.index('State_abbr')))
    df = df[cols]

    return df


# Generate Roads Proportions Data
def get_roads_proportion_data(pas_proxy, lig_proxy, hea_proxy, out_path, proxy_name='Emission Allocation'):
    """
    Formats data for roads proxy emissions
    """
    proxy_name = proxy_name.lower()
    # Add Vehicle Type column
    pas_proxy['Vehicle'] = 'Passenger'
    lig_proxy['Vehicle'] = 'Light'
    hea_proxy['Vehicle'] = 'Heavy'

    # Combine DataFrames
    vmt_roads_proxy = pd.concat([pas_proxy,
                                 lig_proxy,
                                 hea_proxy], axis=0).reset_index(drop=True)
    vmt_roads_proxy = (
        vmt_roads_proxy.rename(columns={'State_abbr': 'state_code',
                                        'Road Type': 'road_type'})
                       .rename(columns=lambda x: str(x).lower())
                       .query("state_code not in ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI', 'UM']")
    )
    vmt_roads_proxy = vmt_roads_proxy[['state_code', 'year', 'vehicle', 'region',
                                       'road_type', proxy_name]]

    vmt_roads_proxy.to_csv(out_path, index=False)

    del pas_proxy, lig_proxy, hea_proxy

    gc.collect()

    return None


def get_road_proxy_data(road_proxy_out_path: Path=task_outputs_path / "roads_proportion_data.csv"):
    if not road_proxy_out_path.exists():
        # Proportional Allocation of Roads Emissions
        # The "Proxy" column in the output data represents the proportion of VMT by vehicle type for each unique combination of urban/rural and road type
        # The following code on the output df results in all 1's:
        # road_proxy_df.groupby(['vehicle', 'state_code', 'year']).Proxy.sum()

        # What this means is that each vehicle type's emissions would need to be allocated at the state/year level
        # What I don't understand is why not use miles travelled if that's the case?
        # The road types from this  output are: Primary, Secondary, and Other
        # These align with TIGER roads classifications

        # Ultimately, I join on two datasets on road type, along with state, year, and region. But this isn't usesful for the proxy value because it's a proportion of VMT across region/road type 

        #################
        # VM2 Outputs
        Miles_road_primary, Miles_road_secondary, Miles_road_other, total, total2 = get_vm2_arrays(num_years)

        # VM4 Outputs
        Per_vmt_mot, Per_vmt_pas, Per_vmt_lig, Per_vmt_hea = get_vm4_arrays(num_years)

        # State Proxy Outputs
        pas_proxy, lig_proxy, hea_proxy, vmt_tot = calculate_state_proxies(num_years,
                                                                        Miles_road_primary,
                                                                        Miles_road_secondary,
                                                                        Miles_road_other,
                                                                        Per_vmt_pas,
                                                                        Per_vmt_lig,
                                                                        Per_vmt_hea)

        # Unpack State Proxy Outputs
        pas_proxy = unpack_state_proxy(pas_proxy, num_years)
        lig_proxy = unpack_state_proxy(lig_proxy, num_years)
        hea_proxy = unpack_state_proxy(hea_proxy, num_years)
        # tot_proxy = unpack_state_allroads_proxy(vmt_tot)

        # Generate Roads Proportions Data
        get_roads_proportion_data(pas_proxy, lig_proxy, hea_proxy, out_path=road_proxy_out_path)

        return pas_proxy, lig_proxy, hea_proxy
    
    return pd.read_csv(road_proxy_out_path)


@benchmark_load
def overlay_cell_state_region(cell_gdf, region_gdf, state_gdf):
    '''
    This function overlays the cell grid with the state and region boundaries for each year of region data
    '''    
    # Overlay the cell grid with the state and region boundaries
    print(f"Overlaying cell grid with state and region boundaries: {datetime.now()}")
    cell_state_region_gdf = gpd.overlay(cell_gdf, state_gdf, how='union')
    print(f"Overlayed cell grid with state boundaries: {datetime.now()}")
    cell_state_region_gdf = gpd.overlay(cell_state_region_gdf, region_gdf, how='union')
    
    # where urban is NaN, set to 0 (since the "region" dataset is urban geometries) and drop variable for year (since it is redundant and in the file name)
    cell_state_region_gdf['urban'] = cell_state_region_gdf['urban'].fillna(0).astype(int)
    cell_state_region_gdf.drop(columns=['year'], inplace=True)

    # drop rows with no cell_id, as this indicates that the geometry falls outside of the US and so doesn't need to be processed
    cell_state_region_gdf.dropna(subset=['cell_id'], inplace=True)

    return cell_state_region_gdf

@benchmark_load
def run_overlay_for_year(year, out_dir):
    # Save the overlaid geoparquet file
    out_path = get_overlay_dir(year, out_dir)
    if out_path.exists():
        print(f"File already exists for {year}: {out_path}")
        return None
    
    # Load the region, cell and state data
    print(f"Reading datasets for {year}: {datetime.now()}")
    cell_gdf = get_cell_gdf().to_crs(4326).reset_index().rename(columns={'index': 'cell_id'})
    region_gdf = get_region_gdf(year)
    state_gdf = get_states_gdf()

    # Overlay the cell grid with the state and region boundaries
    cell_state_region_gdf = overlay_cell_state_region(cell_gdf, region_gdf, state_gdf)

    cell_state_region_gdf.to_parquet(out_path)

    print(f"Saved overlaid geoparquet file for {year} to {out_dir / f'cell_state_region_{year}.parquet'}")