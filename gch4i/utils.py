from pathlib import Path
import calendar

import osgeo  # noqa f401
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
import pandas as pd
import rioxarray  # noqd f401
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.ma as ma
from IPython.display import display
from rasterio.features import rasterize
import warnings

from gch4i.config import global_data_dir_path, figures_data_dir_path
from gch4i.gridding import GEPA_PROFILE, x, y, ARR_SHAPE

Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
Molarch4 = 16.04  # CH4 molecular weight (g/mol)
Res01 = 0.1  # output resolution (degrees)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes
# tg_scale = (
#    0.001  # Tg conversion factor
# )
GWP_CH4 = 25  # global warming potential of CH4 relative to CO2 (used to convert mass
# to CO2e units, from IPPC AR4)
# EEM: add constants (note, we should try to do conversions using variable names, so
#      that we don't have constants hard coded into the scripts)


# TODO: write state / year inventory to GRID allocation, probably using geocube
# EEM: question - for sources where we go from the state down to the grid-level, will
# we still have this calculation step, or will we go straight to the rasterize step?


def state_year_point_allocate_emis(
    proxy_gdf: gpd.GeoDataFrame,
    emi_df: pd.DataFrame,
    proxy_has_year: bool,
    use_proportional: bool,
) -> gpd.GeoDataFrame:
    """
    Allocation state emissions by year to all proxy points within the state
    NOTE:               2024-06-10: only tested with ferroalloys and composting so far.

    Inputs:
    proxy_gdf:    GeoDataFrame: point level proxy data with or without fractional
                        emissions to be used in allocation from state inventory data
    emi_df:             The EPA state level emissions per year, typically read in from
                        the IndDB sheet in the excel workbook
    proxy_has_year:     If the proxy data have a yearly proportional value to use
    use_proportional:   Indicate if the proxy has fractional emissions to be used in the
                        allocation of inventory emissions to the point. For each state /
                        year, the fractional emissions of all points within the state /
                        year are used to allocation inventory emissions.

    Returns:            point_proxy_gdf with new column "allocated_ch4_kt"

    """

    result_list = []
    # for each state and year in the inventory data
    for (state, year), data in emi_df.groupby(["state_code", "year"]):
        # if the proxy has a year, get the proxies for that state / year
        if proxy_has_year:
            state_points = proxy_gdf[
                (proxy_gdf["state_code"] == state) & (proxy_gdf["year"] == year)
            ].copy()
        # else just get the proxy in the state
        else:
            state_points = (
                proxy_gdf[(proxy_gdf["state_code"] == state)].copy().assign(year=year)
            )
        # if there are no proxies in that state, print a warning
        if state_points.shape[0] < 1:
            Warning(f"there are no proxies in {state} but there are emissions!")
            continue
        # get the value of emissions for that state/year
        state_year_emissions = data["ch4_kt"].iat[0]
        # if there are no state inventory emissions, assign all points for that
        # state / year as 0
        if state_year_emissions == 0:
            state_points["allocated_ch4_kt"] = 0
        # else, compute the emission for each proxy record
        else:
            # if the proxy has a proportional value, say from subpart data, use it
            if use_proportional:
                state_points["allocated_ch4_kt"] = (
                    (state_points["ch4_kt"] / state_points["ch4_kt"].sum())
                    * state_year_emissions
                ).fillna(0)
            # else allocate emissions equally to all proxies in state
            else:
                state_points["allocated_ch4_kt"] = (
                    state_year_emissions / state_points.shape[0]
                )
        result_list.append(state_points)
    out_proxy_gdf = pd.concat(result_list).reset_index(drop=True)
    return out_proxy_gdf


def grid_point_emissions(point_gdf):

    area_matrix = load_area_matrix()

    ch4_kt_result_rasters = {}
    ch4_flux_result_rasters = {}
    for year, data in point_gdf.groupby("year"):
        month_days = [calendar.monthrange(year, x)[1] for x in range(1, 13)]
        year_days = np.sum(month_days)

        ch4_kt_raster = rasterize(
            shapes=[
                (shape, value)
                for shape, value in data[["geometry", "allocated_ch4_kt"]].values
            ],
            out_shape=ARR_SHAPE,
            fill=0,
            transform=GEPA_PROFILE["transform"],
            dtype=np.float64,
            merge_alg=rasterio.enums.MergeAlg.add,
        )

        conversion_factor_annual = calc_conversion_factor(year_days, area_matrix)
        ch4_flux_raster = ch4_kt_raster * conversion_factor_annual

        ch4_kt_result_rasters[year] = ch4_kt_raster
        ch4_flux_result_rasters[year] = ch4_flux_raster

    return ch4_kt_result_rasters, ch4_flux_result_rasters


def QC_point_proxy_allocation(proxy_df, emi_df) -> None:
    """take point proxy emi allocations and check against state inventory"""
    print("checking proxy emission allocation by state / year.")
    sum_check = (
        proxy_df.groupby(["state_code", "year"])["allocated_ch4_kt"]
        .sum()
        .reset_index()
        .merge(emi_df, on=["state_code", "year"], how="outer")
        .assign(
            isclose=lambda df: df.apply(
                lambda x: np.isclose(x["allocated_ch4_kt"], x["ch4_kt"]), axis=1
            )
        )
    )

    all_equal = sum_check["isclose"].all()
    print(f"do all proxy emission by state/year equal (isclose): {all_equal}")
    if not all_equal:
        print("states and years with emissions that don't match")
        display(sum_check[~sum_check["isclose"]])

        print(
            (
                "states with no proxy points in them: "
                f"{emi_df[~emi_df['state_code'].isin(proxy_df['state_code'])]['state_code'].unique()}"
            )
        )
        print(
            (
                "states with unaccounted emissions: "
                f"{sum_check[~sum_check['isclose']]['state_code'].unique()}"
            )
        )
    return sum_check


def QC_emi_raster_sums(raster_dict: dict, emi_df: pd.DataFrame) -> None:

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
        emi_df.groupby("year")["ch4_kt"]
        .sum()
        .to_frame()
        .join(gridded_year_sums_df)
        .assign(isclose=lambda df: np.isclose(df["ch4_kt"], df["gridded_sum"]))
    )
    all_equal = sum_check["isclose"].all()

    print(f"do all gridded emission by year equal (isclose): {all_equal}")
    if not all_equal:
        print("if not, these ones below DO NOT equal")
        display(sum_check[~sum_check["isclose"]])
    return sum_check


def calc_conversion_factor(year_days: int, area_matrix: np.array) -> np.array:
    return 10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix


def write_tif_output(in_dict: dict, dst_path: Path) -> None:
    """take an input dictionary with year/array items, write raster to dst_path"""
    out_array = np.stack(list(in_dict.values()))

    dst_profile = GEPA_PROFILE.copy()

    dst_profile.update(count=out_array.shape[0])
    with rasterio.open(dst_path.with_suffix(".tif"), "w", **dst_profile) as dst:
        dst.write(out_array)
        dst.descriptions = [str(x) for x in in_dict.keys()]
    return None


# EEM NOTE: in some scripts we use 0.01x0.01 degree resolution area matrix,
# in other scripts with use the 0.1x0.1 area matrix. Can we add both to this function
# and specify which we're using here?
def load_area_matrix() -> np.array:
    input_path = global_data_dir_path / "gridded_area_m2.tif"
    with rasterio.open(input_path) as src:
        arr = src.read(1)
    return arr


def write_ncdf_output(
    raster_dict: dict,
    dst_path: Path,
    description: str,
    title: str,
    units: str = "moleccm-2s-1",
    # resolution: float,
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

    data_xr = (
        xr.DataArray(
            array_stack,
            coords={
                "time": year_list,
                "lat": y,
                "lon": x,
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
        .rio.write_crs(GEPA_PROFILE["crs"])
        .rio.write_transform(GEPA_PROFILE["transform"])
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        # I think this only write out the year names in a separate table. Doesn't
        # seem useful.
        # .rio.write_coordinate_system()
        .rio.write_nodata(0.0, encoded=True)
    )
    data_xr.to_netcdf(dst_path.with_suffix(".nc"))
    return None


def name_formatter(col: pd.Series):
    """standard name formatted to allow for matching between datasets

    casefold
    replace any repeated spaces with just one
    remove any non-alphanumeric chars

    input:
        col = pandas Series
    returns = pandas series
    """
    return (
        col.str.casefold()
        .str.replace("\s+", " ", regex=True)
        .replace("[^a-zA-Z0-9 -]", "", regex=True)
    )


# %%
def plot_annual_raster_data(ch4_flux_result_rasters, SOURCE_NAME):
    """
    Function to plot the raster data for each year in the dictionary of rasters that are
    output at the end of each sector script.
    """
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

        # Mask the raster data where values are 0. This will make the 0 values
        # transparent and not plotted.
        masked_raster_data = ma.masked_where(raster_data == 0, raster_data)

        # Create a figure and axis with the specified projection
        fig, ax = plt.subplots(
            figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        # Add features to the map
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES)

        # Create a meshgrid for the x and y coordinates
        X, Y = np.meshgrid(x, y)

        # Plot the masked raster data using pcolormesh
        annual_plot = ax.pcolormesh(
            X,
            Y,
            masked_raster_data,
            transform=ccrs.PlateCarree(),
            vmin=10**-15,
            vmax=np.max(raster_data),
            cmap=custom_colormap,
            shading="nearest",
        )

        # Add a colorbar
        plt.colorbar(
            annual_plot,
            ax=ax,
            orientation="horizontal",
            label="Methane emissions (Mg a$^{-1}$ km$^{-2}$)",
        )

        # Set the extent to show the continental US
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.tick_params(labelsize=10)

        # Add a title
        annual_plot_title = f"{year} EPA methane emissions from {SOURCE_NAME}"
        plt.title(annual_plot_title, fontsize=14)

        # Save the plot as a PNG file
        plt.savefig(figures_data_dir_path / f"{SOURCE_NAME}_ch4_flux_{year}.png")

        # Show the plot for review
        plt.show()

        # close the plot
        plt.close()


def plot_raster_data_difference(ch4_flux_result_rasters, SOURCE_NAME):
    """
    Function to plot the difference between the first and last years of the raster data
    for each sector.
    """
    # Generate the plot for the difference between the first and last years.

    # Get the first and last years of the data
    list_of_data_years = list(ch4_flux_result_rasters.keys())
    first_year_data = ch4_flux_result_rasters[list_of_data_years[0]]
    last_year_data = ch4_flux_result_rasters[list_of_data_years[-1]]

    # Calculate the difference between the first and last years
    difference = last_year_data - first_year_data

    # Mask the raster data where values are 0. This will make the 0 values transparent
    # and not plotted.
    difference_masked_raster_data = ma.masked_where(difference == 0, difference)

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
        figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Add features to the map
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)

    # Create a meshgrid for the x and y coordinates
    X, Y = np.meshgrid(x, y)

    # Plot the masked raster data using pcolormesh
    difference_plot = ax.pcolormesh(
        X,
        Y,
        difference_masked_raster_data,
        transform=ccrs.PlateCarree(),
        vmin=10**-15,
        vmax=np.max(difference_masked_raster_data),
        cmap=custom_colormap,
        shading="nearest",
    )

    # Add a colorbar
    plt.colorbar(
        difference_plot,
        ax=ax,
        orientation="horizontal",
        label="Methane emissions (Mg a$^{-1}$ km$^{-2}$)",
    )

    # Set the extent to show the continental US
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.tick_params(labelsize=10)

    # Add a title
    difference_plot_title = f"Difference between {list_of_data_years[0]} and {list_of_data_years[-1]} methane emissions from {SOURCE_NAME}"
    plt.title(difference_plot_title, fontsize=14)

    # Save the plot as a PNG file
    plt.savefig(figures_data_dir_path / f"{SOURCE_NAME}_ch4_flux_difference.png")

    # Show the plot for review
    plt.show()

    # close the plot
    plt.close()
