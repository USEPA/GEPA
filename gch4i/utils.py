import calendar
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
from IPython.display import display
from rasterio.features import rasterize, shapes
from rasterio.plot import show

from gch4i.config import V3_DATA_PATH, figures_data_dir_path, global_data_dir_path
from gch4i.gridding import GEPA_spatial_profile

# import warnings
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


# TODO: a REVERSE FLUX CALCULATION FUNCTION
# def reverse_flux_calculation():               # This function will take the flux and convert it back to emissions (in kt) for a given area
#     pass

# TODO: write state / year inventory to GRID allocation, probably using geocube
# EEM: question - for sources where we go from the state down to the grid-level, will
# we still have this calculation step, or will we go straight to the rasterize step?


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
            time_var = int(time_var)
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
        Warning(f"there is no v2 raster data to compare against v3 for {SOURCE_NAME}!")
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
                percent_dif = 100 * (v3_sum - v2_sum) / ((v3_sum + v2_sum) / 2)
                print(
                    f"year: {year}, v2 flux sum: {v2_sum}, v3 flux sum: {v3_sum}, percent difference: {percent_dif}"
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
                v3_mass_raster = v3_data[year] / conversion_factor_annual
                v3_mass_sum = np.nansum(v3_mass_raster)
                # percent difference between v2 and v3
                percent_dif_mass = (
                    100
                    * (v3_mass_sum - v2_mass_sum)
                    / ((v3_mass_sum + v2_mass_sum) / 2)
                )
                print(
                    f"year: {year}, v2 mass sum: {v2_mass_sum}, v3 mass sum: {v3_mass_sum}, percent difference: {percent_dif_mass}"
                )

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
                difference_plot_title = f"{year} Difference between v2 and v3 methane emissions from {SOURCE_NAME}"
                plt.title(difference_plot_title, fontsize=14)
                # Save the plot as a PNG file
                plt.savefig(
                    str(figures_data_dir_path)
                    + f"/{SOURCE_NAME}_ch4_flux_difference_v2_to_v3_{year}.png"
                )
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
                plt.savefig(
                    str(figures_data_dir_path)
                    + f"/{SOURCE_NAME}_ch4_flux_histogram_{year}.png"
                )
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


def download_url(url, output_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
