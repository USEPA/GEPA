# %%
%load_ext autoreload
%autoreload 2

import pandas as pd
import xarray as xr
from pathlib import Path
import calendar
import rioxarray
import numpy as np
import warnings
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm.auto import tqdm
import seaborn as sns

from gch4i.config import (
    V3_DATA_PATH,
    final_gridded_dir,
    years,
    logging_dir,
)

from gch4i.utils import Avogadro, GEPA_spatial_profile, Molarch4, load_area_matrix
from gch4i.create_final_netcdfs import CreateFinalNetCDFs
import cartopy.crs as ccrs

from gch4i.gridding_utils import GriddingInfo

# %%
file_writer = CreateFinalNetCDFs()
file_writer.write_outputs()
# file_writer.plot_data()
# %%
month_example = file_writer.gch4i_month_scale_dict[2018]
month_example
# %%
annual_example = file_writer.gch4i_flux_dict[2018]
annual_example
# %%
month_example.attrs
# %%
month_example.time.attrs
# %%
month_example["monthly_scale_factor_1A_stationary_combustion"].time.attrs
# %%
month_example["monthly_scale_factor_1A_stationary_combustion"].attrs

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


# Days in year
def get_days_in_year(year):
    year = int(year)
    return 366 if calendar.isleap(year) else 365


# Mass to Flux Conversion Factor
def calc_conversion_factor(year_days: int, area_matrix):
    """calculate emissions in kt to flux (in units of molec. cm-2 s-1)"""
    return 10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix
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
        area_matrix = ds["grid_cell_area"].squeeze(dim="time", drop=True)  # final area matrix within netcdf file
        # area_matrix = area_matrix.sel(lat=area_matrix.lat[::-1])
        flux_data = ds[f"emi_ch4_{igroup}"]
        mass_data = flux_data / calc_conversion_factor(iyear_days, area_matrix)
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
    # %%

    # # Plot final flux data
    # v3_arr = np.array(list(flux_data_dict.values()))
    # final_flux_data_arr = xr.DataArray(
    #     v3_arr,
    #     dims=["time", "y", "x"],
    #     coords={
    #         "time": years,
    #         "y": gepa_profile.y,
    #         "x": gepa_profile.x,
    #     },
    #     name=igroup,
    # )
    v3_arr = xr.merge(list(flux_data_dict.values())) #, compat="equals")

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

    # ADD PLOTTING CONVERSION FACTOR

    final_flux_data_arr = convert_flux_for_plotting(final_flux_data_arr)

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


# HL started writing a class but felt like it was repeating a lot of existing code.
# The above code works to get the QC results but needs to be added to create_final_netcdfs.py
class QCFinalNetCDFs:
    def __init__(self, group_name):
        # the path to the directory where the final gridded data is saved
        self.group_name = group_name
        self.flux_data_files = list(final_gridded_dir.glob("*AugTest.nc"))
        self.gepa_profile = GEPA_spatial_profile()
        self.qc_dir = logging_dir

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


# %%
# # get the object needed to manage the gridding operations
# g_info = GriddingInfo(update_mapping=True, save_file=True)
# group_names = list(g_info.v2_df['gch4i_name'])

# for igroup in group_names:
#     file_checker = QCFinalNetCDFs(igroup)
#     file_checker.qc_national_mass_sums()
#     # file_checker.qc_plot_flux_maps()
# %%


from gch4i.config import prelim_gridded_dir
from gch4i.utils import load_area_matrix, years
import rasterio

# %%
def plot_monthly_scaling():
    scale_dict = {}
    for x, year in tqdm(enumerate(years), total=len(years), desc="Processing years"):
        print(x, year)
        in_path = (
            final_gridded_dir
            / f"Gridded_GHGI_Methane_v3_Monthly_Scale_Factors_{year}_draft.nc"
        )
        scale_dict[year] = xr.open_dataset(in_path)
    all_scale_datasets = xr.concat(list(scale_dict.values()), dim="time").drop_vars(
        "spatial_ref"
    )
    all_scale_datasets

    all_scale_summary_list = []
    for var in all_scale_datasets:
        if var == "spatial_ref":
            continue
        tmp_ds = all_scale_datasets[var]

        month_plot_df = (
            tmp_ds.to_dataframe()
            .reset_index()
            .assign(month=lambda x: x.time.dt.month, year=lambda x: x.time.dt.year)
            .rename(columns={var: "monthly_flux"})
            .dropna(subset=["monthly_flux"])
            .query("monthly_flux != 0")
            .assign(source=var)
        )
        all_scale_summary_list.append(month_plot_df)
    all_scale_summary_df = pd.concat(all_scale_summary_list, ignore_index=True)
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
    plt.show()
    plt.savefig(f"{logging_dir}/all_final_monthly_scaling.png")
    plt.close()
# %%
def plot_a_year(in_ds_path):

    in_ds = xr.open_dataset(in_ds_path)
    in_ds.close()
    in_ds

    ncol = 4
    nrow = len(in_ds) // ncol + 1
    ncol, nrow

    width = 30
    height = width * 1.6

    fig, axs = plt.subplots(nrow, ncol, figsize=(width, height), dpi=300)

    for var, ax in zip(in_ds, axs.ravel()):
        # print(in_ds[var].attrs)
        try:
            tmp_da = in_ds[var]
            # print(tmp_da)
            fg = (
                tmp_da.where(lambda x: x != 0)
                .isel(time=0)
                .plot.imshow(
                    # col="time",
                    # col_wrap=3,
                    cmap="magma",
                    # cmap=self.emi_custom_colormap,
                    # transform=ccrs.PlateCarree(),  # remember to provide this!
                    # subplot_kws={"projection": ccrs.PlateCarree()},
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "shrink": 0.5,
                        "aspect": 40,
                        "extend": "neither",
                        # "label": "methane emissions (Mg a$^{-1}$ km$^{-2}$)",
                    },
                    robust=True,
                    # figsize=(20, 20),
                    ax=ax,
                )
            )
            ax.set(title=var)
            ax.set_axis_off()
        except:
            print(f"issue with {var}")
    fig.suptitle(f"final flux for year {year}")
    fig.tight_layout()
    plt.show()


# %%

for year in years:
    in_path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{year}_AugTest.nc"
    plot_a_year(in_path)
# plot_a_year(v3_scale_file)

# %%
for var in v3_scale_ds:
    if var == "spatial_ref":
        continue
    tmp_ds = v3_scale_ds[var]
    p = tmp_ds.where(lambda x: x != 0).plot(
        transform=ccrs.PlateCarree(),
        col="time",
        col_wrap=4,
        cmap="magma",
        subplot_kws={"projection": ccrs.Orthographic(-80, 35)},
    )
    for ax in p.axs.flat:
        ax.coastlines()
        ax.gridlines()
    plt.show()
# %%
# For reference, we can look at the attributes (and other features) of the v2 data
v2_flux_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_2012.nc"
v2_scale_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_Monthly_Scale_Factors_2012.nc"
# %%
v2_scale_ds = xr.open_dataset(v2_scale_file)
v2_scale_ds.close()
v2_scale_ds
# %%
v2_flux_ds = xr.open_dataset(v2_flux_file)
v2_flux_ds.close()
v2_flux_ds
# %%
gepa_profile = GEPA_spatial_profile()
year = 2020
input_flux_paths = list(prelim_gridded_dir.glob("*ch4_emi_flux.tif"))
for in_path in input_flux_paths[:1]:
    source_cat = in_path.stem.split("_")[0]
    name_parts = in_path.stem.split("_")[:-3]
    long_name = f"""{year} Methane emissions from IPCC source category {' '.join(name_parts)}"""
    var_name = f"emi_ch4_{'_'.join(name_parts)}"
    group_ds = (
        xr.open_dataset(in_path)
        .sel(band=0 + 1)
        .rename(
            {"band_data": var_name, "band": "time", "x": "lon", "y": "lat"}
        )
        .expand_dims({"time": 1})
        .assign_coords(
            {
                "time": [0.0],
                "lon": gepa_profile.x,
                # "lat": gepa_profile.y,
                "lat": np.flip(gepa_profile.y),
            }
        )
        .set_coords(["time", "lon", "lat"])
        .reset_coords("spatial_ref", drop=True)
    )
    group_ds.isel(time=0)[var_name].plot.imshow()

# %%
group_ds.lat
# %%
group_ds.lon

# %%
area_ds = (
            load_area_matrix()
            .expand_dims(dim={"time": 1})
            .rename({"x": "lon", "y": "lat"})
            .to_dataset(name="grid_cell_area")
        )
area_ds
# %%
out_ds = xr.merge([group_ds, area_ds], compat="equals")
out_ds
# %%
area_ds.lon.values == group_ds.lon.values
# %%
out_ds.isel(time=0)[var_name].plot.imshow()
# %%
out_ds.isel(time=0)["grid_cell_area"].plot.imshow()
# %%
flux_data_dict = {}
for iyear in years:
    in_path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{iyear}_AugTest.nc"
    ds = xr.open_dataset(in_path).drop_vars("spatial_ref").assign_coords(time=[iyear])
    flux_data_dict[iyear] = ds
    ds.close()
# %%
all_flux_ds = xr.merge(list(flux_data_dict.values()))
# %%
for in_ds in flux_data_dict.values():
    print(year)
    for var in in_ds:
        print("\t", var, in_ds[var].dims == ('time', 'lat', 'lon'), in_ds[var].shape == (1, 350, 700))
# %%
all_flux_ds = xr.concat(flux_data_dict.values(), dim="time")
# %%
for var in all_flux_ds:
    print(var)
    tmp_ds = all_flux_ds[var]
    tmp_ds.where(lambda x: x != 0).plot.imshow(
        col="time",
        col_wrap=4,
    )
# %%
tmp_ds
# %%
get_days_in_year(2012)
# %%
for year in all_flux_ds["time"].values:
    print(year)
    tmp_ds = all_flux_ds.sel(time=year)
    for var in tmp_ds:
        print("\t", var, tmp_ds[var].dims == ('lat', 'lon'), tmp_ds[var].shape == (350, 700))
# %%
tmp_ds
# %%
area_ds = tmp_ds["grid_cell_area"]
area_ds = area_ds.sel(lat=area_ds.lat[::-1])
emi_ds = tmp_ds.drop_vars("grid_cell_area")
area_ds.plot()
plt.show()
emi_ds["emi_ch4_1A_stationary_combustion"].plot()
plt.show()
# %%
days_in_this_year = [get_days_in_year(x) for x in years]
conversion_list = [calc_conversion_factor(days, area_ds) for days in days_in_this_year]
conversion_ds = xr.concat(conversion_list, dim="time").assign_coords(time=years)
conversion_ds
# %%
mass_ds = tmp_ds.drop_vars("grid_cell_area") / conversion_ds
# %%
mass_ds
# %%
mass_ds["emi_ch4_1A_stationary_combustion"].isel(time=0).where(lambda x: x != 0).plot.imshow()
# %%
mass_ds.sum(dim=["lat", "lon"]).to_dataframe()
# %%
