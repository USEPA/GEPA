"""
Name:                   1B1a_energy_abandoned_coal.py
Date Last Modified:     2024-06-07
Authors Name:           N. Kruskamp, H. Lohman (RTI International), Erin McDuffie
                        (EPA/OAP)
Purpose:                Spatially allocates methane emissions for source category 1B1a,
                        sector energy, source Abandoned Underground Coal Mines.
Input Files:            - 
Output Files:           - 
Notes:                  - 
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------

# for testing/development
# %load_ext autoreload
# %autoreload 2
from zipfile import ZipFile

import osgeo  # noqa
import duckdb
import geopandas as gpd
import pandas as pd
import seaborn as sns
from geopy.geocoders import Nominatim
from IPython.display import display

from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import (
    QC_emi_raster_sums,
    QC_point_proxy_allocation,
    grid_point_emissions,
    name_formatter,
    plot_annual_raster_data,
    plot_raster_data_difference,
    state_year_point_allocate_emis,
    tg_to_kt,
    write_ncdf_output,
    write_tif_output,
)

# from pytask import Product, task

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)


# %%
def get_liberated_an_coal_inv_data(input_path):
    """read in the ch4_kt values for each state"""

    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=115,
            usecols="A:BA",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # drop columns we don't need
        # # get just methane emissions
        .query("(ghg == 'CH4') & (subcategory1.str.contains('Liberated'))")
        .drop(
            columns=[
                "sector",
                "category",
                "subcategory1",
                "subcategory2",
                "subcategory3",
                "subcategory4",
                "subcategory5",
                "carbon pool",
                "fuel1",
                "fuel2",
                "exclude",
                "id",
                "sensitive (y or n)",
                "data type",
                "subsector",
                "crt code",
                "units",
                "ghg",
                "gwp",
            ]
        )
        # set the index to state
        .rename(columns={"georef": "state_code"})
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
    )
    return emi_df


def get_rec_and_used_an_coal_inv_data(input_path):
    """read in the ch4_kt values for each state"""

    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=115,
            usecols="A:BA",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # drop columns we don't need
        # # get just methane emissions
        .query("(ghg == 'CH4') & (subcategory1.str.contains('Recovered &Used'))")
        .drop(
            columns=[
                "sector",
                "category",
                "subcategory1",
                "subcategory2",
                "subcategory3",
                "subcategory4",
                "subcategory5",
                "carbon pool",
                "fuel1",
                "fuel2",
                "exclude",
                "id",
                "sensitive (y or n)",
                "data type",
                "subsector",
                "crt code",
                "units",
                "ghg",
                "gwp",
            ]
        )
        # set the index to state
        .rename(columns={"georef": "state_code"})
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
    )
    return emi_df
    # %%


# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = "1B1a"
SECTOR_NAME = "Energy"
SOURCE_NAME = "Abandoned Underground Coal Mines"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
)

# PATHS
sector_dir = V3_DATA_PATH / "sector/abandoned_mines"

# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# INVENTORY INPUT FILE
inventory_workbook_path = ghgi_data_dir_path / "AbandonedCoalMines1990-2022_FRv1.xlsx"

# PROXY INPUT FILES
# state vector refence with state_code
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
# input 1: NOTE: mine list comes from inventory workbook
# input 2: abandoned mines address list
# downloaded manually from:
# https://www.msha.gov/data-and-reports/mine-data-retrieval-system
# scroll down to section "Explore MSHA Datasets"
# select from drop down: "13 Mines Data Set"
msha_path = sector_dir / "Mines.zip"


# %% STEP 1. Load GHGI-Proxy Mapping Files

# %% STEP 2: Read In EPA State GHGI Emissions by Year ----------------------------------

# Get state vectors and state_code for use with inventory and proxy data
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

EPA_state_liberated_emi_df = get_liberated_an_coal_inv_data(inventory_workbook_path)
EPA_state_randu_emi_df = get_rec_and_used_an_coal_inv_data(inventory_workbook_path)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_liberated_emi_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    # legend=False,
)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_randu_emi_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    # legend=False,
)

# %% STEP 3: GET AND FORMAT PROXY DATA -------------------------------------------------
"""
Location information (e.g., latitude/longitude) for each mine is taken from the MSHA
database.

Mines in the AMM Opportunities database are matched to the MSHA databased
based first on 

1. MSHA ID. In total, 430 mines are matched based on MSHA ID,


If there is no match on the ID then the mine is attempted to be
matched based on the mine...

# NOTE: I have had better luck without the date for the intial match after ID
2. name, state, county, and abandonment date.



1 based on name, date, county, and state,

6 on date, state, and county, and

6 based on mine name, state, and county out of a total 544 mines.


For the remaining 101 mines, these emissions are allocated across the county on record
for that mine (based on the area in each grid cell relative to the total area in that
county). This approach is also used for 6 mines where the reported Lat/Lon in the MSHA
database does not match the state on record for that mine. In total, ~20% of total
annual abandoned mine emissions are allocated to the county-level rather than specific
mine location. 
"""

# %%
county_raw = gpd.read_file(
    "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"
)
# %%
county_gdf = (
    county_raw.rename(columns=str.lower)
    .astype({"statefp": int})
    .merge(state_gdf[["state_code", "statefp"]], on="statefp")
    .assign(
        formatted_county=lambda df: name_formatter(df["name"]),
        formatted_state=lambda df: name_formatter(df["state_code"]),
    )
    .to_crs(4326)
)
county_gdf


# %%
def get_ab_coal_mine_proxy_data():
    pass


with ZipFile(msha_path) as z:
    with z.open("Mines.txt") as f:
        msha_df = (
            pd.read_table(
                f,
                sep="|",
                encoding="ISO-8859-1",
                # usecols=["MINE_ID", "LATITUDE", "LONGITUDE"],
            )
            # XXX: what does this do? (from v2 notebook)
            .query("COAL_METAL_IND == 'C'")
            .dropna(subset=["LATITUDE", "LONGITUDE"])
            .assign(
                formatted_name=lambda df: name_formatter(df["CURRENT_MINE_NAME"]),
                formatted_county=lambda df: name_formatter(df["FIPS_CNTY_NM"]),
                formatted_state=lambda df: name_formatter(df["STATE"]),
                date_abd=lambda df: pd.to_datetime(df["CURRENT_STATUS_DT"]),
            )
            # .set_index("MINE_ID")
        )

msha_gdf = gpd.GeoDataFrame(
    msha_df.drop(columns=["LATITUDE", "LONGITUDE"]),
    geometry=gpd.points_from_xy(msha_df["LONGITUDE"], msha_df["LATITUDE"]),
    crs=4326,
)
msha_gdf = msha_gdf[msha_gdf.intersects(state_gdf.dissolve().geometry.iat[0])]
# %%
ax = msha_gdf.plot(color="xkcd:scarlet", figsize=(10, 10))
state_gdf.boundary.plot(ax=ax, color="xkcd:slate", lw=0.2, zorder=1)
# %%

msha_cols = [
    "MINE_ID",
    "geometry",
    "CURRENT_MINE_STATUS",
    "CURRENT_CONTROLLER_BEGIN_DT",
]


ghgi_mine_list = (
    pd.read_excel(
        inventory_workbook_path,
        sheet_name="Mine List",
        skiprows=1,
        # nrows=115,
        usecols="B:M",
    )
    # name column names lower
    .rename(columns=lambda x: str(x).lower())
    .assign(
        MINE_ID=lambda df: pd.to_numeric(df["msha id"], errors="coerce")
        .fillna(0)
        .astype(int),
        formatted_name=lambda df: name_formatter(df["mine name"]),
        formatted_county=lambda df: name_formatter(df["county"]),
        formatted_state=lambda df: name_formatter(df["state"]),
        date_abd=lambda df: pd.to_datetime(df["date of aban."]),
    )
    .set_index("MINE_ID")
    .join(msha_gdf[msha_cols].set_index("MINE_ID"))
    .reset_index()
)
ghgi_mine_list
# %%

# First try to match by ID alone
print(f"total mines: {ghgi_mine_list.shape[0]}\n")
matches = ghgi_mine_list[~ghgi_mine_list["geometry"].isna()]
missing_geoms = ghgi_mine_list[ghgi_mine_list["geometry"].isna()]
print(f"mines matched by ID: {matches.shape[0]}")
print(f"mines still missing geom: {missing_geoms.shape[0]}\n")

# Then try to match by a combination of columns in each dataset
matching_attempts = {
    "name, date, county, state": [
        "formatted_name",
        "date_abd",
        "formatted_county",
        "formatted_state",
    ],
    "name, county, state": ["formatted_name", "formatted_county", "formatted_state"],
    "date, county, state": ["date_abd", "formatted_county", "formatted_state"],
    "name, date": ["formatted_name", "date_abd"],
}

match_result_list = []
match_result_list.append(matches)
for match_name, col_list in matching_attempts.items():
    print(missing_geoms.shape)

    # match_attempt = missing_geoms.drop(columns=["geometry"]).merge(
    match_attempt = missing_geoms.drop(columns=msha_cols).merge(
        # msha_gdf.drop_duplicates(subset=col_list)[col_list + ["geometry"]],
        msha_gdf.drop_duplicates(subset=col_list)[col_list + msha_cols],
        # msha_gdf[col_list + ["geometry"]],
        left_on=col_list,
        right_on=col_list,
        how="left",
    )
    print(match_attempt.shape)
    matches = match_attempt[~match_attempt["geometry"].isna()]
    missing_geoms = match_attempt[
        match_attempt["geometry"].isna()
    ]  # .drop(columns="MINE_ID")
    match_result_list.append(matches)
    print(f"mines matched by {match_name}: {matches.shape[0]}")
    print(f"mines still missing geom: {missing_geoms.shape[0]}\n")

match_attempt = missing_geoms.drop(columns="geometry").merge(
    county_gdf[["formatted_county", "formatted_state", "geometry"]],
    left_on=["formatted_county", "formatted_state"],
    right_on=["formatted_county", "formatted_state"],
    how="left",
)
matches = match_attempt[~match_attempt["geometry"].isna()]
missing_geoms = match_attempt[match_attempt["geometry"].isna()]
match_result_list.append(matches)
print(f"mines matched by county and state to county polygon: {matches.shape[0]}")
print(f"mines still missing geom: {missing_geoms.shape[0]}\n")

matching_results = pd.concat(match_result_list)
print(f"total geo match mines: {matching_results.shape[0]}\n")

# %%


# %% STEP 3.1: QA/QC proxy data --------------------------------------------------------

# calculate fraction of year mine operated during closing year if it closed during
# our study period

# it appears the new workbooks has a clean version of 'Current Emissions Status' called
# "simple status" so there is no need to recode that from the broader list found in v2.

# 2B) Clean up recovery status text

## Step 3 - Add basin number. #CA, IL, NA, BW, WS
basin_name_recode = {
    "Central Appl.": 0,
    "Illinois": 1,
    "Northern Appl.": 2,
    "Warrior": 3,
    "Uinta": 4,
    "Raton": 4,
    "Arkoma": 4,
    "Piceance": 4,
}

b_medium = np.array([2.329011, 2.314585, 2.292595, 2.299685, 2.342465])
D_sealed = np.array([0.000741, 0.000733, 0.000725, 0.000729, 0.000747])
D_venting = np.array([0.003735, 0.003659, 0.003564, 0.003601, 0.003803])
D_flooded = np.array([0.672, 0.672, 0.672, 0.672, 0.672])

# %%
import numpy as np

calc_results = matching_results.assign(
    recovering=lambda df: np.where(
        df["current model worksheet"].str.casefold().str.contains("recovering"), 1, 0
    ),
    Basin_nr=lambda df: df["coal basin"].replace(basin_name_recode),
    reopen_date=lambda df: pd.to_datetime(df["CURRENT_CONTROLLER_BEGIN_DT"]),
    days_closed=lambda df: (df["reopen_date"] - df["date_abd"]).dt.days.where(lambda s: s > 0, 0),
)

# %%

# matching_results[matching_results["CURRENT_MINE_STATUS"].eq("Active")][
#     ["MINE_ID", "CURRENT_CONTROLLER_BEGIN_DT", "date_abd", "days_closed"]
# ]
calc_results[
    ["MINE_ID", "reopen_date", "date_abd"]
]

calc_results[(calc_results["reopen_date"] > calc_results["date_abd"])][
    ["MINE_ID", "reopen_date", "date_abd", "days_closed"]
]

# there are 21 mines that have reopened during the study period that we will need to
# to handle differently with partial year emissions calculations
calc_results[(calc_results["reopen_date"].dt.year > 2012)][
    ["MINE_ID", "reopen_date", "date_abd", "days_closed"]
]


# %%
## Step 4. Record the number of days since the mine has closed
# 2024-06-11: left off here. I think I can calculate the days since closure and then
# the proxy calculations by year, format into a long table, and then processing would
# be similar to ferroalloys in cycling through facilities by state / year with a
# proportional value to allocate emissions
import datetime

days_since_close_list = {}
for year in range(min_year, max_year + 1):
    matching_results[f"days_closed_{year}"] = (
        datetime.datetime(year=year, month=7, day=2) - matching_results["date_abd"]
    )

# TODO: I think we're going to need to add the two sources together eventually
# that is giong to require a rework of the gridding function probably to separate out
# the flux calculation to after that:
# 1: calc first source
# 2: calc second source
# 3: math them together
# 4: calc flux
# 5: QC / save / plot as normal?


# %%
# timeseries look at the data.
g = sns.lineplot(
    ab_mine_proxy_gdf,
    x="year",
    y="ch4_kt",
    hue="facility_name",
    legend=True,
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 0.9))
sns.despine()
# some checks of the data
# how many na values are there
print("Number of NaN values:")
display(ab_mine_proxy_gdf.isna().sum())
# how many missing locations are there by year
print("Number of Facilities with Missing Locations Each Year")
display(ab_mine_proxy_gdf[ab_mine_proxy_gdf.is_empty]["year"].value_counts())
# how many missing locations are there by facility name
print("For Each Facility with Missing Data, How Many Missing Years")
display(
    ab_mine_proxy_gdf[ab_mine_proxy_gdf.is_empty]["formatted_fac_name"].value_counts()
)
# a plot of the timeseries of emission by facility
sns.lineplot(
    data=ab_mine_proxy_gdf, x="year", y="ch4_kt", hue="facility_name", legend=False
)

# %% MAP PROXY DATA --------------------------------------------------------------------
ax = ab_mine_proxy_gdf.drop_duplicates("formatted_fac_name").plot(
    "formatted_fac_name", categorical=True, cmap="Set2", figsize=(10, 10)
)
state_gdf.boundary.plot(ax=ax, color="xkcd:slate", lw=0.2, zorder=1)

# %% STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES ---------------------------
allocated_emis_gdf = state_year_point_allocate_emis(
    ab_mine_proxy_gdf,
    EPA_state_liberated_emi_df,
    proxy_has_year=True,
    use_proportional=True,
)
allocated_emis_gdf

# %% STEP 4.1: QC PROXY ALLOCATED EMISSIONS BY STATE AND YEAR --------------------------
proxy_qc_result = QC_point_proxy_allocation(
    allocated_emis_gdf, EPA_state_liberated_emi_df
)

sns.relplot(
    kind="line",
    data=proxy_qc_result,
    x="year",
    y="allocated_ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

sns.relplot(
    kind="line",
    data=EPA_state_liberated_emi_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)
proxy_qc_result

# %% STEP 5: RASTERIZE THE CH4 KT AND FLUX ---------------------------------------------
ch4_kt_result_rasters, ch4_flux_result_rasters = grid_point_emissions(
    allocated_emis_gdf
)

# %% STEP 5.1: QC GRIDDED EMISSIONS BY YEAR --------------------------------------------
# TODO: report QC metrics for flux values compared to V2: descriptive statistics
qc_kt_rasters = QC_emi_raster_sums(ch4_kt_result_rasters, EPA_state_liberated_emi_df)
qc_kt_rasters

# %% STEP 6: SAVE THE FILES ------------------------------------------------------------
write_tif_output(ch4_kt_result_rasters, ch4_kt_dst_path)
write_tif_output(ch4_flux_result_rasters, ch4_flux_dst_path)
write_ncdf_output(
    ch4_flux_result_rasters,
    ch4_flux_dst_path,
    netcdf_title,
    netcdf_description,
)

# %% STEP 7: PLOT THE RESULTS AND DIFFERENCE, SAVE FIGURES TO FILES --------------------
plot_annual_raster_data(ch4_flux_result_rasters, SOURCE_NAME)
plot_raster_data_difference(ch4_flux_result_rasters, SOURCE_NAME)

# %%
