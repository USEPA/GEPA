# %%
# %load_ext autoreload
# %autoreload 2

import calendar
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import seaborn as sns
import xarray as xr
from scipy.constants import Avogadro
from tqdm.auto import tqdm

from gch4i.config import V3_DATA_PATH, final_gridded_dir, logging_dir, years
from gch4i.create_final_netcdfs import CreateFinalNetCDFs
from gch4i.gridding_utils import GriddingInfo, GroupGridder
from gch4i.utils import GEPA_spatial_profile, Molarch4

# %%
file_writer = CreateFinalNetCDFs()
file_writer.write_outputs()
# %%
g_info = GriddingInfo(update_mapping=True, save_file=True)
group_names = list(g_info.v2_df["gch4i_name"])
# %%
# The EPA color map from their V2 plots
gepa_profile = GEPA_spatial_profile()
emi_custom_colormap = colors.LinearSegmentedColormap.from_list(
    name="emi_cmap",
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


# HL started writing a class but felt like it was repeating a lot of existing code.
# The above code works to get the QC results but needs to be added to create_final_netcdfs.py
class QCFinalNetCDFs:
    def __init__(self, group_name):
        # the path to the directory where the final gridded data is saved
        self.group_name = group_name
        self.flux_data_files = list(final_gridded_dir.glob("*AugTest.nc"))
        self.gepa_profile = GEPA_spatial_profile()
        self.qc_dir = logging_dir

    convert_flux_for_plotting = GroupGridder.convert_flux_for_plotting
    get_days_in_year = GroupGridder.get_days_in_year
    get_days_in_month = GroupGridder.get_days_in_month
    calc_conversion_factor = GroupGridder.calc_conversion_factor
    convert_flux_for_plotting = GroupGridder.convert_flux_for_plotting

    def qc_national_mass_sums(self):
        v3_data_dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for out_path in self.flux_data_files:
                v3_year = int(out_path.stem.split("_")[-2])
                # v3_year = int(out_path.stem.split("_")[-1])
                final_group_name = "emi_ch4_" + self.group_name
                v3_data = rioxarray.open_rasterio(out_path, variable=final_group_name)[
                    final_group_name
                ].values.squeeze(axis=0)
                self.v3_area_matrix = rioxarray.open_rasterio(
                    out_path, variable="grid_cell_area"
                )["grid_cell_area"].values.squeeze(axis=0)
                v3_data_dict[v3_year] = v3_data

        v3_arr = np.array(list(v3_data_dict.values()))

        self.v3_flux_da = xr.DataArray(
            v3_arr,
            dims=["time", "y", "x"],
            coords=[
                list(v3_data_dict.keys()),
                self.gepa_profile.y,
                self.gepa_profile.x,
            ],
            name=final_group_name,
        )

        def calculate_mass(in_ds):
            """calculates mass for dictionary of total flux year/array pairs"""
            times = in_ds.time.values

            def get_days_in_year(year):
                year = int(year)
                return 366 if calendar.isleap(year) else 365

            def calc_conversion_factor(
                year_days: int, area_matrix: np.array
            ) -> np.array:
                """calculate emissions in kt to flux (in units of molec. cm-2 s-1)"""
                return (
                    10**9
                    * Avogadro
                    / float(Molarch4 * year_days * 24 * 60 * 60)
                    / area_matrix
                )

            days_in_year = [get_days_in_year(x) for x in times]
            conv_factors = [
                calc_conversion_factor(x, self.v3_area_matrix) for x in days_in_year
            ]

            conv_ds = xr.DataArray(
                conv_factors,
                # np.flip(conv_factors, 1),
                dims=["time", "y", "x"],
                coords=[times, self.gepa_profile.y, self.gepa_profile.x],
                name="conversion_factor",
            )
            flux_out_da = in_ds / conv_ds
            return flux_out_da

        self.v3_mass_da = calculate_mass(self.v3_flux_da)
        v3_mass_yearly_sums = np.nansum(self.v3_mass_da.values, axis=(1, 2))

        self.mass_sums_df = pd.DataFrame(
            {
                "year": list(v3_data_dict.keys()),
                "v3_sum": v3_mass_yearly_sums,
            }
        ).assign(metric="mass")

        # Save mass_sums_df to the qc folder for the group
        self.mass_sums_df.to_csv(
            self.qc_dir
            / f"{self.group_name}/{self.group_name}_ch4_v3_mass_sum_final_gridded_data_qc.csv",
            index=False,
        )
        return self.mass_sums_df

    def qc_plot_flux_maps(self) -> None:
        """
        Function to plot the raster data for each year in the dictionary of rasters that are
        output at the end of each sector script.
        """

        # we set 0 and negative values as NA

        # apply the conversion factor to the annual flux data for plotting
        plotting_data = self.v3_flux_da.where(lambda x: x != 0)
        # print(plotting_data.groupby("time").max(dim=...).values)
        plotting_data = xr.where(plotting_data > 10, 10, plotting_data)
        # print(plotting_data.groupby("time").max(dim=...).values)
        fg = plotting_data.plot.imshow(
            col="time",
            col_wrap=3,
            cmap=self.emi_custom_colormap,
            transform=ccrs.PlateCarree(),  # remember to provide this!
            subplot_kws={"projection": ccrs.PlateCarree()},
            # interpolation=None,
            vmin=10**-15,
            vmax=10,
            cbar_kwargs={
                "orientation": "horizontal",
                "shrink": 0.8,
                "aspect": 40,
                "extend": "neither",
                "label": "methane emissions (Mg a$^{-1}$ km$^{-2}$)",
            },
            robust=True,
            figsize=(20, 20),
        )

        for ax in fg.axs.ravel():
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES)
            ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())

        fg.fig.suptitle(
            f"{self.group_name}\nGridded methane flux emissions", fontsize=14
        )

        # Save the plots as PNG files to the figures directory
        plt.savefig(
            self.qc_dir
            / f"{self.group_name}_ch4_v3_annual_flux_final_gridded_data_qc.png"
        )
        # Show the plot for review
        plt.show()
        # close the plot
        plt.close()


def get_all_scale_data():
    scale_dict = {}
    for year in tqdm(years, total=len(years), desc="reading scale files"):
        in_path = (
            final_gridded_dir
            / f"Gridded_GHGI_Methane_v3_Monthly_Scale_Factors_{year}_draft.nc"
        )
        in_ds = xr.open_dataset(in_path)
        scale_dict[year] = in_ds
        in_ds.close()
    out_ds = xr.concat(list(scale_dict.values()), dim="time").drop_vars("spatial_ref")
    return out_ds


def get_all_flux_data():
    flux_data_dict = {}
    for iyear in tqdm(years, desc="reading flux files"):
        in_path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{iyear}_AugTest.nc"
        ds = (
            xr.open_dataset(in_path)
            .drop_vars("spatial_ref")
            .assign_coords(time=[iyear])
        )
        flux_data_dict[iyear] = ds
        ds.close()

    out_ds = xr.merge(list(flux_data_dict.values()))
    return out_ds


def plot_a_year(in_ds_path):

    year = in_ds_path.name.split("_")[4]

    in_ds = xr.open_dataset(in_ds_path)
    in_ds.close()
    in_ds

    ncol = 4
    nrow = len(in_ds) // ncol + len(in_ds) % ncol
    ncol, nrow

    width = 30
    height = width * 1.6

    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(width, height),
        dpi=300,
        subplot_kw=dict(projection=ccrs.Orthographic(-95.75, 36.75)),
    )

    for var, ax in zip(in_ds, axs.ravel()):
        # print(in_ds[var].attrs)
        if var == "spatial_ref":
            continue
        try:
            tmp_da = in_ds[var]
            # print(var)

            (
                tmp_da.where(lambda x: x != 0)
                .isel(time=0)
                .plot.imshow(
                    cmap="magma",
                    transform=ccrs.PlateCarree(),
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "shrink": 0.5,
                        "aspect": 40,
                        "extend": "neither",
                    },
                    robust=True,
                    ax=ax,
                )
            )
            ax.coastlines()
            ax.gridlines()
            ax.add_feature(cfeature.STATES)
            ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
            ax.set(title=var)
            ax.set_axis_off()
        except Exception as e:
            print(f"issue with {var}: {e}")
    fig.suptitle(f"final flux for year {year}")
    fig.tight_layout()
    plt.savefig(f"{logging_dir}/all_final_flux_{year}.png")
    # plt.show()
    plt.close()


def plot_group_flux_maps(in_ds):

    for var in in_ds:
        print(var)
        tmp_ds = in_ds[var]
        tmp_ds.where(lambda x: x != 0).plot.imshow(
            col="time",
            col_wrap=4,
        )
        plt.show()


def plot_monthly_scaling(in_ds):
    all_scale_summary_df = (
        in_ds.where(lambda x: x != 0)
        .mean(dim=["lat", "lon"])
        .to_dataframe()
        .reset_index()
        .melt(id_vars=["time"], var_name="source", value_name="monthly_flux")
        .dropna(subset=["monthly_flux"])
        .assign(
            year=lambda x: x.time.dt.year,
            month=lambda x: x.time.dt.month,
        )
    )

    all_scale_summary_df
    g = sns.relplot(
        kind="line",
        data=all_scale_summary_df,
        col="source",
        col_wrap=4,
        x="month",
        y="monthly_flux",
        hue="year",
        palette="tab20",
        height=6,
        aspect=2,
        facet_kws={"sharey": False, "sharex": True},
    )
    plt.savefig(logging_dir / "all_final_monthly_scaling.png")
    plt.show()
    plt.close()


def plot_group_scale_maps(in_ds):
    for var in in_ds:
        print(var)
        tmp_ds = in_ds[var]
        tmp_ds.where(lambda x: x != 0).plot.imshow(
            col="time", col_wrap=12, cmap="magma"
        )
        plt.show()


# Days in year
def get_days_in_year(year):
    year = int(year)
    return 366 if calendar.isleap(year) else 365


# Mass to Flux Conversion Factor
def calc_conversion_factor(year_days: int, area_matrix):
    """calculate emissions in kt to flux (in units of molec. cm-2 s-1)"""
    return 10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix
    # apply the conversion factor to the annual flux data for plotting


def convert_flux_for_plotting(flux_da: xr.DataArray) -> xr.DataArray:
    """
    Convert the flux data from molec/cm2/s to Mg/km2/year for plotting.
    This is a helper function to be used in the plotting methods.

    This required the input data array to have a time dimension repping years.
    """

    res_list = []
    for i, time in enumerate(flux_da.time.values):
        year = pd.to_datetime(time).year
        year_days = get_days_in_year(year)
        res = (
            flux_da.sel(time=time)
            / float(10**6 * Avogadro)
            * (year_days * 24 * 60 * 60)
            * Molarch4
            * float(1e10)
        )
        res_list.append(res)

    out_ds = xr.concat(res_list, dim="time")

    return out_ds


def plot_original_scale_figs():

    scaling_files = list(logging_dir.rglob("*_monthly_scaling.png"))
    scaling_files = [f for f in scaling_files if not f.name.startswith("all")]
    sorted(scaling_files)
    _, axs = plt.subplots(4, 4, figsize=(10, 10))
    for in_path, ax in zip(scaling_files, axs.ravel()):
        img = mpimg.imread(in_path)
        ax.imshow(img)
        ax.axis("off")
    plt.show()


# %%

for igroup in group_names:
    # Calculate the total national emissions by source and year
    target_mass_sum_list_path: Path = (
        logging_dir / f"{igroup}/{igroup}_ch4_v3_emi_qc.csv"
    )
    target_mass_sum_df = pd.read_csv(target_mass_sum_list_path)["ghgi_ch4_kt"]
    mass_sum_list = []
    flux_arr_list = []
    flux_data_dict = {}
    for iyear in years:
        iyear_days = get_days_in_year(iyear)  # number of days in the year
        # print(iyear, iyear_days)
        final_data_path: Path = (
            final_gridded_dir / f"Gridded_GHGI_Methane_v3_{iyear}_AugTest.nc"
        )  # path to final gridded data
        ds = xr.open_dataset(final_data_path)  # final gridded netcdf file
        ds.close()
        area_matrix = ds["grid_cell_area"].squeeze(
            dim="time", drop=True
        )  # final area matrix within netcdf file
        # area_matrix = area_matrix.sel(lat=area_matrix.lat[::-1])
        flux_data = ds[f"emi_ch4_{igroup}"]
        conv_arr = calc_conversion_factor(iyear_days, area_matrix)
        mass_data = flux_data / conv_arr
        mass_sum = np.nansum(mass_data)
        mass_sum_list.append(mass_sum)
        flux_data_dict[iyear] = flux_data.sel(time=0.0)
    mass_sum_df = pd.DataFrame(
        {
            "year": years,
            "final_sum": mass_sum_list,
            "target_sum": target_mass_sum_df,
        }
    ).assign(
        isclose_pass=lambda df: np.isclose(
            df["final_sum"], df["target_sum"], atol=0.0, rtol=0.0001
        )
    )

    print(igroup, mass_sum_df["isclose_pass"].all())
    mass_sum_df.to_csv(
        logging_dir / f"{igroup}/{igroup}_ch4_v3_mass_sum_final_gridded_data_qc.csv",
        index=False,
    )

    v3_arr = xr.concat(flux_data_dict.values(), dim="time").assign_coords(time=years)

    # ADD PLOTTING CONVERSION FACTOR

    final_flux_data_arr = convert_flux_for_plotting(v3_arr)

    plotting_data = final_flux_data_arr.where(lambda x: x != 0)
    # print(plotting_data.groupby("time").max(dim=...).values)
    plotting_data = xr.where(plotting_data > 10, 10, plotting_data)
    # print(plotting_data.groupby("time").max(dim=...).values)
    fg = plotting_data.plot.imshow(
        col="time",
        col_wrap=3,
        cmap=emi_custom_colormap,
        transform=ccrs.PlateCarree(),  # remember to provide this!
        subplot_kws={"projection": ccrs.PlateCarree()},
        # interpolation=None,
        vmin=10**-15,
        vmax=10,
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.8,
            "aspect": 40,
            "extend": "neither",
            "label": "methane emissions (Mg a$^{-1}$ km$^{-2}$)",
        },
        robust=True,
        figsize=(20, 20),
    )

    for ax in fg.axs.ravel():
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES)
        ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())

    fg.fig.suptitle(f"{igroup}\nGridded methane flux emissions", fontsize=14)

    # Save the plots as PNG files to the figures directory
    plt.savefig(
        logging_dir / f"{igroup}/{igroup}_ch4_v3_annual_flux_final_gridded_data_qc.png"
    )
    # Show the plot for review
    plt.show()
    # close the plot
    plt.close()

# %%


# %%
# # get the object needed to manage the gridding operations
# g_info = GriddingInfo(update_mapping=True, save_file=True)
# group_names = list(g_info.v2_df['gch4i_name'])

# for igroup in group_names:
#     file_checker = QCFinalNetCDFs(igroup)
#     file_checker.qc_national_mass_sums()
#     # file_checker.qc_plot_flux_maps()

# %%
all_flux_ds = get_all_flux_data()
all_flux_ds
# %%

scaling_ds = get_all_scale_data()
scaling_ds


# %%


plot_monthly_scaling(scaling_ds)


# %%
for year in years:
    in_path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{year}_AugTest.nc"
    plot_a_year(in_path)


plot_group_scale_maps(scaling_ds)
# %%

plot_group_flux_maps(all_flux_ds)
# %%


# %%
# For reference, we can look at the attributes (and other features) of the v2 data
# v2_flux_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_2012.nc"
# v2_scale_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_Monthly_Scale_Factors_2012.nc"

# v2_scale_ds = xr.open_dataset(v2_scale_file)
# v2_scale_ds.close()
# v2_scale_ds

# v2_flux_ds = xr.open_dataset(v2_flux_file)
# v2_flux_ds.close()
# v2_flux_ds
# %%
