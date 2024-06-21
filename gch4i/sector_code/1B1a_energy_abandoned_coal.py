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
import calendar
import datetime

import osgeo  # noqa
import duckdb
import geopandas as gpd
import pandas as pd
import seaborn as sns
from geopy.geocoders import Nominatim
from IPython.display import display
import numpy as np

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
    QC_proxy_allocation,
    grid_allocated_emissions,
    name_formatter,
    plot_annual_raster_data,
    plot_raster_data_difference,
    allocate_emissions_to_proxy,
    tg_to_kt,
    write_ncdf_output,
    write_tif_output,
    calculate_flux,
    combine_gridded_emissions,
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
inventory_workbook_path = (
    ghgi_data_dir_path / "coal/AbandonedCoalMines1990-2022_FRv1.xlsx"
)

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
# TODO: finalize this function after review
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

# get a crosswalk of records that we can't find via the mine DB or county, develop
# a crosswalk for the 3 records we can't find a match for from these "county" names
# to actual county names.
# https://en.wikipedia.org/wiki/West,_West_Virginia
# https://en.wikipedia.org/wiki/Dickenson_County,_Virginia
# Finding the county for rosedale is a bit tricky. It's not a formal place?
# https://www.mapquest.com/us/tennessee/rosedale-tn-282922764
county_name_fixes = {
    "formatted_county": {
        "dickerson": "dickenson",
        "west": "wetzel",
        "rosedale": "anderson",
    }
}

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
    .rename(columns={"active emiss. (mmcfd)": "active_emiss", "state": "state_code"})
    .assign(
        MINE_ID=lambda df: pd.to_numeric(df["msha id"], errors="coerce")
        .fillna(0)
        .astype(int),
        formatted_name=lambda df: name_formatter(df["mine name"]),
        formatted_county=lambda df: name_formatter(df["county"]),
        formatted_state=lambda df: name_formatter(df["state_code"]),
        date_abd=lambda df: pd.to_datetime(df["date of aban."]),
    )
    .replace(county_name_fixes)
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
    # match_attempt = missing_geoms.drop(columns=["geometry"]).merge(
    match_attempt = missing_geoms.drop(columns=msha_cols).merge(
        # msha_gdf.drop_duplicates(subset=col_list)[col_list + ["geometry"]],
        msha_gdf.drop_duplicates(subset=col_list)[col_list + msha_cols],
        # msha_gdf[col_list + ["geometry"]],
        left_on=col_list,
        right_on=col_list,
        how="left",
    )

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
print("mines that are in the abandoned workbook but listed as active in the mine db:")
matching_results[matching_results["CURRENT_MINE_STATUS"] == "Active"]
# %%
# %% STEP 3.1: QA/QC proxy data --------------------------------------------------------

# calculate fraction of year mine operated during closing year if it closed during
# our study period

# it appears the new workbooks has a clean version of 'Current Emissions Status' called
# "simple status" so there is no need to recode that from the broader list found in v2.

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

# basin coefficients pulled from v2:
basin_coef_dict = {
    "Flooded": [0.672, 0.672, 0.672, 0.672, 0.672],
    "Sealed": [0.000741, 0.000733, 0.000725, 0.000729, 0.000747],
    "Venting": [0.003735, 0.003659, 0.003564, 0.003601, 0.003803],
    "b_medium": [2.329011, 2.314585, 2.292595, 2.299685, 2.342465],
}

basin_coef_df = pd.DataFrame.from_dict(basin_coef_dict)
basin_coef_df
# %%
calc_results = matching_results.assign(
    recovering=lambda df: np.where(
        df["current model worksheet"].str.casefold().str.contains("recovering"), 1, 0
    ),
    basin_nr=lambda df: df["coal basin"].replace(basin_name_recode),
    reopen_date=lambda df: pd.to_datetime(df["CURRENT_CONTROLLER_BEGIN_DT"]),
).loc[
    :,
    [
        "MINE_ID",
        "geometry",
        "state_code",
        "county",
        "basin_nr",
        "recovering",
        "reopen_date",
        "date_abd",
        "simple status",
        "active_emiss",
    ],
]
calc_results


# %%
# TODO: Current workbook is missing some years of this data. update this when we get
# the final workbook. wrap this up into a single table that we query by year in the
# loop below.
ratios_df = (
    pd.read_excel(inventory_workbook_path, sheet_name="2018", skiprows=20, nrows=5)
    .loc[:, ["Sealed %", "Vented %", "Flooded %"]]
    .rename(
        columns={"Sealed %": "Sealed", "Vented %": "Venting", "Flooded %": "Flooded"}
    )
)  # .apply(lambda x: x / x.sum(), axis=1)
# ).loc[:, ["Basin", "Sealed %", "Vented %", "Flooded %"]]

ratios_normed_df = ratios_df / ratios_df.sum().sum()
ratios_normed_df
# %%
# # orig flooded calc
# abdmines["Active Emiss. (mmcfd) "][imine] * np.exp(-1 * D_flooded[ibasin] * ab_numyrs)
# # orig venting calc
# abdmines["Active Emiss. (mmcfd) "][imine] * (
#     1 + b_medium[ibasin] * D_venting[ibasin] * ab_numyrs
# ) ** (-1 / float(b_medium[ibasin]))
# # orig sealed calc
# abdmines["Active Emiss. (mmcfd) "][imine] * (1 - 0.8) * (
#     1 + b_medium[ibasin] * D_sealed[ibasin] * ab_numyrs
# ) ** (-1 / float(b_medium[ibasin]))


def flooded_calc(df, basin, bc_df):
    return df["active_emiss"] * np.exp(
        -1 * bc_df.loc[basin, "Flooded"] * df["years_closed"]
    )


def venting_calc(df, basin, bc_df):
    return df["active_emiss"] * (
        1
        + bc_df.loc[basin, "b_medium"]
        * bc_df.loc[basin, "Venting"]
        * df["years_closed"]
    ) ** (-1 / bc_df.loc[basin, "b_medium"])


def sealed_calc(df, basin, bc_df):
    return (
        df["active_emiss"]
        * (1 - 0.8)
        * (
            1
            + bc_df.loc[basin, "b_medium"]
            * bc_df.loc[basin, "Sealed"]
            * df["years_closed"]
        )
        ** (-1 / bc_df.loc[basin, "b_medium"])
    )


def unknown_calc(df, basin, bc_df, r_df):
    em_flo = r_df.loc[basin, "Flooded"] * flooded_calc(df, basin, bc_df)
    em_ven = r_df.loc[basin, "Venting"] * venting_calc(df, basin, bc_df)
    em_sea = r_df.loc[basin, "Sealed"] * sealed_calc(df, basin, bc_df)
    return np.sum([em_flo, em_ven, em_sea])


result_list = []
for year in range(min_year, max_year + 1):

    # get the year days
    month_days = [calendar.monthrange(year, x)[1] for x in range(1, 13)]
    year_days = np.sum(month_days)

    # this year date to calc relative emissions
    calc_date = datetime.datetime(year=year, month=7, day=2)

    year_df = (
        calc_results.copy()
        # use only mines that are closed this year or earlier
        # e.g. if it's 2012 and the mine closed in 2022, remove it because it is not
        # abandoned yet.
        .query("date_abd.dt.year <= @year")
        .query("active_emiss > 0")
        .assign(
            year=year,
            # calculate the number of days closed relative to 07/02 of this year?
            # XXX: why not calculate the entire year?
            # if the mine closed this year, give it special treatment where the number of
            # days closed is 1/2 of the number of days closed relative to our date of 07/02
            # the logic here is: if the mine closed in this year, the number of days closed
            # is equal to 1/2 the days closed relative to 07/02.
            days_closed=lambda df: np.where(
                df["date_abd"].dt.year == year,
                -((calc_date - df["date_abd"]) / 2),
                calc_date - df["date_abd"],
            ),
            years_closed=lambda df: (df["days_closed"].dt.days / year_days),
            emis_mmcfd=0,
        )
    )
    print(year, year_df.shape[0])

    data_list = []
    for (basin, status), data in year_df.groupby(["basin_nr", "simple status"]):
        if status == "Flooded":
            data["emis_mmcfd"] = flooded_calc(data, basin, basin_coef_df)
        if status == "Venting":
            data["emis_mmcfd"] = venting_calc(data, basin, basin_coef_df)
        if status == "Sealed":
            data["emis_mmcfd"] = sealed_calc(data, basin, basin_coef_df)
        if status == "Unknown":
            data["emis_mmcfd"] = unknown_calc(
                data, basin, basin_coef_df, ratios_normed_df
            )
        data_list.append(data)
    year_df = pd.concat(data_list)
    year_df.loc[year_df["recovering"].eq(1), "emis_mmcfd"] = 0
    result_list.append(year_df)

result_df = pd.concat(result_list)
result_df["year"].value_counts().sort_index()
# %%
sns.relplot(data=result_df, y="emis_mmcfd", x="year", hue="state_code", kind="line")
sns.relplot(data=result_df, y="emis_mmcfd", x="year", kind="line")
# %%
result_df.groupby(["year", "state_code"])["emis_mmcfd"].sum()
# %%
result_df.groupby("year")["emis_mmcfd"].sum().describe()
# %%
# looking at values against the v2 numbers, it's not far off in some cases, but in
# others it's a drastic difference. This needs more input and review. We're close, but
# either the estimates have changed or more likely I've messed up somewhere...
# coefficients by year are wrong here so maybe a factor?
# Could be worth running this code against the v2 workbook to see if I get the same
# answer.
result_df[result_df.year == 2018].groupby("state_code")["emis_mmcfd"].sum()
ab_coal_proxy_gdf = gpd.GeoDataFrame(result_df, crs=4326)
# NOTE: This ends the get proxy function, so once reviewed, clean it up.

# %% STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES ---------------------------
allocated_liberated_emis_gdf = allocate_emissions_to_proxy(
    ab_coal_proxy_gdf,
    EPA_state_liberated_emi_df,
    proxy_has_year=True,
    use_proportional=True,
    proportional_col_name="emis_mmcfd",
)

allocated_randu_emis_gdf = allocate_emissions_to_proxy(
    ab_coal_proxy_gdf,
    EPA_state_randu_emi_df,
    proxy_has_year=True,
    use_proportional=True,
    proportional_col_name="emis_mmcfd",
)

# %% STEP 4.1: QC PROXY ALLOCATED EMISSIONS BY STATE AND YEAR --------------------------
proxy_qc_liberated_result = QC_proxy_allocation(
    allocated_liberated_emis_gdf, EPA_state_liberated_emi_df
)
proxy_qc_liberated_result

# %%
proxy_qc_randu_result = QC_proxy_allocation(
    allocated_randu_emis_gdf, EPA_state_randu_emi_df
)
proxy_qc_randu_result

# %% STEP 5: RASTERIZE THE CH4 KT AND FLUX ---------------------------------------------
ch4_kt_liberated_result_rasters = grid_allocated_emissions(allocated_liberated_emis_gdf)

ch4_kt_randu_result_rasters = grid_allocated_emissions(allocated_randu_emis_gdf)

# %% STEP 5.1: QC GRIDDED EMISSIONS BY YEAR --------------------------------------------
# TODO: report QC metrics for flux values compared to V2: descriptive statistics
qc_kt_liberated_rasters = QC_emi_raster_sums(
    ch4_kt_liberated_result_rasters, EPA_state_liberated_emi_df
)
qc_kt_liberated_rasters
# %%
qc_kt_randu_rasters = QC_emi_raster_sums(
    ch4_kt_randu_result_rasters, EPA_state_randu_emi_df
)
qc_kt_randu_rasters

# %% STEP 5.2: COMBINE SUBSOURCE RASTERS TOGETHER --------------------------------------
ch4_kt_result_rasters = combine_gridded_emissions(
    [ch4_kt_liberated_result_rasters, ch4_kt_randu_result_rasters]
)
ch4_flux_result_rasters = calculate_flux(ch4_kt_result_rasters)

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