# %%
%load_ext autoreload
%autoreload 2
# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr

from gch4i.config import V3_DATA_PATH, prelim_gridded_dir
from gch4i.gridding_utils import GriddingInfo

sns.set_theme(style="darkgrid")
# %%


def get_v2_data():
    v2_data_paths = V3_DATA_PATH.glob("Gridded_GHGI_Methane_v2_*.nc")
    v2_data_paths = [f for f in v2_data_paths if "Monthly_Scale_Factors" not in f.stem]
    v2_data_dict = {}

    for in_path in v2_data_paths:
        v2_year = int(in_path.stem.split("_")[-1])
        v2_data = xr.open_dataset(in_path)
        v2_data_dict[v2_year] = v2_data
    v2_ds = (
        xr.merge(v2_data_dict.values())
        .sortby(["time", "lat", "lon"])
        .drop_vars("grid_cell_area")
        .fillna(0)
    )

    return v2_ds


def get_v3_data():
    v3_data_dict = {}
    for row in v3_v2_data_df.itertuples():
        v2_key = row.v2_key
        v3_key = row.gch4i_name
        print(v2_key, v3_key)
        if not v2_key:
            print(f"Skipping {v3_key} as it has no v2_key")
            print()
            continue
        v3_data_path = prelim_gridded_dir / f"{v3_key}_ch4_emi_flux.tif"
        print(v3_key, v3_data_path.exists())
        v3_data = (
            xr.open_dataset(v3_data_path)
            .drop_vars("spatial_ref")
            .rename({"band_data": v2_key})
        )
        v3_data_dict[v3_key] = v3_data
        print()
    v3_ds = (
        xr.merge(v3_data_dict.values())
        .rename({"band": "time", "x": "lon", "y": "lat"})
        .isel(time=slice(0, 7))
        .sortby(["time", "lat", "lon"])
        .fillna(0)  # Fill NaNs with 0
    )

    return v3_ds


def align_versions(v2_ds, v3_ds):
    """
    Aligns the v2 and v3 datasets by their coordinates.
    """
    v3_ds = v3_ds.assign_coords(time=v2_ds.time, lat=v2_ds.lat, lon=v2_ds.lon)
    v2_ds, v3_ds = xr.align(v2_ds, v3_ds, join="exact")
    return v2_ds, v3_ds


def get_group_crosswalk():
    g_info = GriddingInfo()
    g_info.display_all_group_statuses()
    v3_v2_data_df = g_info.ready_groups_df[["gch4i_name", "v2_key"]].drop_duplicates()
    return v3_v2_data_df


# %%
# get the emi/proxy data guide, status data, and v2/v3 crosswalk data


v3_v2_data_df = get_group_crosswalk()
v3_v2_data_df

# %%
v2_ds = get_v2_data()
v2_ds
# %%
v3_ds = get_v3_data()
v3_ds
# %%
v2_ds, v3_ds = align_versions(v2_ds, v3_ds)
# %%
flux_dif_ds = v3_ds - v2_ds
flux_dif_ds
# %%
flux_dif_df = (
    flux_dif_ds.to_dataframe()
    .dropna(how="all")
    .reset_index()
    .drop(columns=["lat", "lon"])
    .melt(id_vars=["time"], var_name="gch4i_name", value_name="flux_diff")
    .dropna(subset=["flux_diff"])
    .query("flux_diff != 0")
    .assign(time=lambda x: x.time.dt.year.astype(int))
)
flux_dif_df
# %%

sns.relplot(
    data=flux_dif_df,
    x="time",
    y="flux_diff",
    kind="line",
    col="gch4i_name",
    col_wrap=3,
    errorbar="sd",
    legend=True,
    facet_kws={"sharey": False, "sharex": True},
)

# %%
sns.relplot(
    data=flux_dif_df.query(
        "gch4i_name == 'emi_ch4_5D_Wastewater_Treatment_Industrial'"
    ),
    x="time",
    y="flux_diff",
    kind="line",
    errorbar="sd",
)

# %%
for gch4i_name, group_df in flux_dif_df.groupby("gch4i_name"):
    print(gch4i_name)
    print(group_df.groupby("time").flux_diff.mean())
    print()
# %%
flux_dif_ds["emi_ch4_5D_Wastewater_Treatment_Industrial"].where(lambda x: x != 0).plot(
    col="time",
    col_wrap=3,
    cmap="coolwarm",
)
# %%


def plot_group_flux(group_name, v2_ds, v3_ds):
    v2_tmp = (
        v2_ds[group_name]
        .to_dataframe()
        .reset_index()
        # .drop(columns=["lat", "lon"])
        .rename(columns={group_name: "v2_flux"})
        # .query("v2_flux != 0")
        .dropna(subset=["v2_flux"])
        .assign(time=lambda x: x.time.dt.year.astype(int))
    )
    v2_tmp

    v3_tmp = (
        v3_ds[group_name]
        .to_dataframe()
        .reset_index()
        # .drop(columns=["lat", "lon"])
        .rename(columns={group_name: "v3_flux"})
        # .query("v3_flux != 0")
        .dropna(subset=["v3_flux"])
        .assign(time=lambda x: x.time.dt.year.astype(int))
    )
    v3_tmp

    compare_df = (
        v2_tmp.merge(
            v3_tmp, on=["time", "lat", "lon"], how="outer", suffixes=("_v2", "_v3")
        )
        .dropna(subset=["v2_flux", "v3_flux"])
        .melt(
            id_vars=["time", "lat", "lon"],
        )
    )
    compare_df

    sns.lineplot(
        data=compare_df,
        x="time",
        y="value",
        hue="variable",
        # kind="scatter",
        # kind="line",
        # col_wrap=3,
        # height=4,
        # aspect=1.5,
    )
    return None


plot_group_flux("emi_ch4_5B1_Composting", v2_ds, v3_ds)
# %%
for group_name in v3_v2_data_df.v2_key.dropna().unique():
    print(group_name)
    plot_group_flux(group_name, v2_ds, v3_ds)
    plt.show()
    print()


# %%


def plot_all_group_fluxes(v2_ds, v3_ds):

    v2_tmp = (
        v2_ds.to_dataframe()
        .reset_index()
        .drop(columns=["lat", "lon"])
        .assign(time=lambda x: x.time.dt.year.astype(int))
        .melt(
            id_vars=["time"],
        )
        .assign(version="v2")
        .query("value != 0")
        .dropna(subset=["value"])
    )

    v3_tmp = (
        v3_ds.to_dataframe()
        .reset_index()
        .drop(columns=["lat", "lon"])
        .assign(time=lambda x: x.time.dt.year.astype(int))
        .melt(
            id_vars=["time"],
        )
        .assign(version="v3")
        .query("value != 0")
        .dropna(subset=["value"])
    )
    v3_tmp

    compare_df = pd.concat([v2_tmp, v3_tmp])
    compare_df

    sns.relplot(
        data=compare_df,
        x="time",
        y="value",
        hue="version",
        kind="line",
        col="variable",
        col_wrap=3,
        # errorbar="sd",
        legend=True,
        palette="bright",
        facet_kws={"sharey": False, "sharex": True},
    )
    return None


# plot_all_group_fluxes(v2_ds, v3_ds)
# %%


from scipy.constants import Avogadro
from gch4i.utils import Molarch4
import numpy as np
import calendar

# %%
years = pd.to_datetime(v2_ds.time.values).year.astype(int)
years
# %%


def cal_cov_factor(year):
    if calendar.isleap(year):
        year_days = 366
    else:
        year_days = 365
    conv_factor = (
        float(10**6 * Avogadro) * (year_days * 24 * 60 * 60) * Molarch4 * float(1e10)
    )
    return conv_factor


conv_ds = xr.Dataset(
    {
        "conversion_factor": (("time"), [cal_cov_factor(year) for year in years]),
    },
    coords={
        "time": v2_ds.time,
    },
)
# %%

res_list = []
for i, time in enumerate(v3_ds.time.values):
    year = pd.to_datetime(time).year
    if calendar.isleap(year):
        year_days = 366
    else:
        year_days = 365
    print(year, year_days)
    res = v3_ds.sel(time=time) / float(10**6 * Avogadro) * (year_days * 24 * 60 * 60) * Molarch4 * float(1e10)
    res_list.append(res)

v3_converted_ds = xr.concat(res_list, dim="time")
v3_converted_ds
# %%



v2_converted_ds = v2_ds / conv_ds.conversion_factor
v3_converted_ds = v3_ds / conv_ds.conversion_factor


# %%

v3_converted_ds["emi_ch4_3A_Enteric_Fermentation"].where(lambda x: x != 0).plot(
    col="time",
    col_wrap=3,
    cmap="coolwarm",
)


# %%
plot_all_group_fluxes(v2_converted_ds, v3_converted_ds)
# %%

from matplotlib import colors, ticker
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
import numpy as np


def plot_annual_emission_flux_map(in_ds, title_str="map", scale_max=10):
    # Define constants
    Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
    Molarch4 = 16.04  # g/mol
    month_day_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_day_nonleap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    Lat, Lon, year_range, Emi_flux_map = (
        in_ds.lat.values,
        in_ds.lon.values,
        in_ds.time.dt.year.values,
        in_ds.values,
    )

    print(Emi_flux_map.shape)


    converted_data = []
    for iyear in np.arange(len(year_range)):
        if year_range[iyear] == 2012 or year_range[iyear] == 2016:
            year_days = np.sum(month_day_leap)
        else:
            year_days = np.sum(month_day_nonleap)
        # my_cmap = copy(plt.cm.get_cmap('rainbow',lut=3000))
        # my_cmap._init()
        # slopen = 200
        # alphas_slope = np.abs(np.linspace(0, 1.0, slopen))
        # alphas_stable = np.ones(3003-slopen)
        # alphas = np.concatenate((alphas_slope, alphas_stable))
        # my_cmap._lut[:,-1] = alphas
        # my_cmap.set_under('gray', alpha=0)

        ##Rainbow:
        my_cmap = colors.LinearSegmentedColormap.from_list(
            name="my_cmap",
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
        my_cmap._init()
        slopen = 200
        alphas_slope = np.abs(np.linspace(0, 1.0, slopen))
        alphas_stable = np.ones(3003 - slopen)
        alphas = np.concatenate((alphas_slope, alphas_stable))
        my_cmap._lut[:, -1] = alphas
        my_cmap.set_under("gray", alpha=0)

        Lon_cor = Lon[50:632] - 0.05
        Lat_cor = Lat[43:300] - 0.05

        xpoints = Lon_cor
        ypoints = Lat_cor
        yp, xp = np.meshgrid(ypoints, xpoints)

        if np.shape(Emi_flux_map)[0] == len(year_range):
            zp = Emi_flux_map[iyear, 43:300, 50:632]
            # zp = Emi_flux_map[iyear, :, :]
        elif np.shape(Emi_flux_map)[2] == len(year_range):
            zp = Emi_flux_map[43:300, 50:632, iyear]
            # zp = Emi_flux_map[:, :, iyear]

        # zp = (
        #     zp
        #     / (float(10**6 * Avogadro)
        #     * (year_days * 24 * 60 * 60)
        #     * Molarch4
        #     * float(1e10))
        # )
        zp = (
            zp
            / float(10**6 * Avogadro)
            * (year_days * 24 * 60 * 60)
            * Molarch4
            * float(1e10)
        )
        converted_data.append(zp)

        fig, ax = plt.subplots(dpi=300)
        m = Basemap(
            llcrnrlon=xp.min(),
            llcrnrlat=yp.min(),
            urcrnrlon=xp.max(),
            urcrnrlat=yp.max(),
            projection="merc",
            # resolution="h",
            area_thresh=5000,
        )
        m.drawmapboundary(fill_color="Azure")
        m.fillcontinents(color="FloralWhite", lake_color="Azure", zorder=1)
        m.drawcoastlines(linewidth=0.5, zorder=3)
        m.drawstates(linewidth=0.25, zorder=3)
        m.drawcountries(linewidth=0.5, zorder=3)

        xpi, ypi = m(xp, yp)
        plot = m.pcolor(
            xpi,
            ypi,
            zp.transpose(),
            cmap=my_cmap,
            vmin=10**-15,
            vmax=scale_max,
            snap=True,
            zorder=2,
        )
        cb = m.colorbar(plot, location="bottom", pad="1%")
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        cb.ax.set_xlabel("Methane emissions (Mg a$^{-1}$ km$^{-2}$)", fontsize=10)
        cb.ax.tick_params(labelsize=10)
        Titlestring = str(year_range[iyear]) + " " + title_str
        fig1 = plt.gcf()
        plt.title(Titlestring, fontsize=14)
        plt.show()
    return converted_data


# %%
# plot_annual_emission_flux_map(v2_ds["emi_ch4_3A_Enteric_Fermentation"])

# %%
v3_converted_data = plot_annual_emission_flux_map(v3_ds["emi_ch4_3A_Enteric_Fermentation"])

# %%
v3_converted_ds_from_orig = xr.Dataset(
    {
        "emi_ch4_3A_Enteric_Fermentation": (
            ("time", "lat", "lon"),
            v3_converted_data,
        ),
    },
    coords={
        "time": v3_ds.time,
        "lat": v3_ds.lat[43:300],
        "lon": v3_ds.lon[50:632],
    },
)
v3_converted_ds_from_orig
# %%
plot_all_group_fluxes(v3_converted_ds_from_orig, v3_converted_ds["emi_ch4_3A_Enteric_Fermentation"])
# %%
v3_converted_ds_from_orig["emi_ch4_3A_Enteric_Fermentation"].where(lambda x: x!=0).groupby("time").max(dim=...).values
# %%
v3_converted_ds["emi_ch4_3A_Enteric_Fermentation"].where(lambda x: x!=0).groupby("time").max(dim=...).values
# %%
