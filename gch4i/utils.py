from pathlib import Path

import osgeo  # noqa f401
import numpy as np
import rasterio
import xarray as xr
import pandas as pd
import rioxarray  # noqd f401
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy.ma as ma

from gch4i.config import global_data_dir_path, figures_data_dir_path
from gch4i.gridding import GEPA_PROFILE, x, y

Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
Molarch4 = 16.04  # CH4 molecular weight (g/mol)
Res01 = 0.1  # output resolution (degrees)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes
# tg_scale = (
#    0.001  # Tg conversion factor
# )
GWP_CH4 = 25  # global warming potential of CH4 relative to CO2 (used to convert mass to CO2e units, from IPPC AR4)
# EEM: add constants (note, we should try to do conversions using variable names, so
#      that we don't have constants hard coded into the scripts)


def calc_conversion_factor(days_in_year: int, cell_area_matrix: np.array) -> float:
    return (
        10**9
        * Avogadro
        / float(Molarch4 * days_in_year * 24 * 60 * 60)
        / cell_area_matrix
    )


def write_tif_output(in_dict: dict, dst_path: Path) -> None:
    """take an input dictionary with year/array items, write raster to dst_path"""
    out_array = np.stack(list(in_dict.values()))

    dst_profile = GEPA_PROFILE.copy()

    dst_profile.update(count=out_array.shape[0])
    with rasterio.open(dst_path, "w", **dst_profile) as dst:
        dst.write(out_array)
        dst.descriptions = [str(x) for x in in_dict.keys()]
    return None


def load_area_matrix() -> np.array:
    input_path = global_data_dir_path / "gridded_area_m2.tif"
    with rasterio.open(input_path) as src:
        arr = src.read(1)
    return arr


# %%
def write_ncdf_output(
    raster_dict: dict,
    dst_path: Path,
    title: str,
    description: str,
    # resolution: float,
    units: str = "moleccm-2s-1",
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
    data_xr.to_netcdf(dst_path)
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
def plot_annual_raster_data(ch4_flux_result_rasters):
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

        # Mask the raster data where values are 0. This will make the 0 values transparent and not plotted.
        masked_raster_data = ma.masked_where(raster_data == 0, raster_data)

        # Create a figure and axis with the specified projection
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

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
        annual_plot_title = f"{year} EPA methane emissions from {SECTOR_NAME.split('_')[-1]} production"
        plt.title(annual_plot_title, fontsize=14)

        # Show the plot for review
        plt.show()

        # Save the plot as a PNG file
        plt.savefig(str(figures_data_dir_path) + f"/{SECTOR_NAME}_ch4_flux_{year}.png")

        # close the plot
        plt.close()


def plot_raster_data_difference(ch4_flux_result_rasters):
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

    # Mask the raster data where values are 0. This will make the 0 values transparent and not plotted.
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
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

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
    difference_plot_title = f"Difference between {list_of_data_years[0]} and {list_of_data_years[-1]} methane emissions from {SECTOR_NAME.split('_')[-1]} production"
    plt.title(difference_plot_title, fontsize=14)

    # Show the plot for review
    plt.show()

    # Save the plot as a PNG file
    plt.savefig(str(figures_data_dir_path) + f"/{SECTOR_NAME}_ch4_flux_difference.png")

    # close the plot
    plt.close()