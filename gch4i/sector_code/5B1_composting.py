# %%
# Name: 5A1_landfills.py

# Authors Name: N. Kruskamp, H. Lohman (RTI International), Erin McDuffie (EPA/OAP)
# Date Last Modified: 5/21/2024
# Purpose: Spatially allocates methane emissions for source category 2C2 Ferroalloy production
#
# Input Files:
#      - State_Ferroalloys_1990-2021.xlsx, SubpartK_Ferroalloy_Facilities.csv,
#           all_ghgi_mappings.csv, all_proxy_mappings.csv
# Output Files:
#      - f"{INDUSTRY_NAME}_ch4_kt_per_year.tif, f"{INDUSTRY_NAME}_ch4_emi_flux.tif"
# Notes:
# TODO: update to use facility locations from 2024 GHGI state inventory files
# TODO: include plotting functionaility
# TODO: include netCDF writting functionality

# ---------------------------------------------------------------------
# %% STEP 0. Load packages, configuration files, and local parameters

import calendar
import warnings
from pathlib import Path

import osgeo  # noqa
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import seaborn as sns
from IPython.display import display
import fiona

# from pytask import Product, task
from rasterio.features import rasterize
from shapely import wkb

# TODO: use dotenv .env file to load local paths
from gch4i.config import (
    data_dir_path,
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.gridding import ARR_SHAPE, GEPA_PROFILE
from gch4i.utils import (
    calc_conversion_factor,
    load_area_matrix,
    name_formatter,
    tg_to_kt,
    write_ncdf_output,
    write_tif_output,
)

# gpd.io.file.fiona.drvsupport.supported_drivers['kml'] = 'rw'
# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# gpd.io.file.fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
# fiona.drvsupport.supported_drivers['kml'] = 'rw'  # enable KML support which is disabled by default
# fiona.drvsupport.supported_drivers['KML'] = 'rw'  # enable KML support which is disabled by default
# fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'  # enable KML support which is disabled by default
gpd.options.io_engine = "pyogrio"

# %%

# the NAICS code pulled from the v2 notebook for facilities in the FRS data
COMPOSTING_FRS_NAICS_CODE = 562219
SECTOR_NAME = "5B1_waste_composting"
DUPLICATION_TOLERANCE_M = 500

ch4_kt_dst_path = tmp_data_dir_path / f"{SECTOR_NAME}_ch4_kt_per_year.tif"
ch4_flux_dst_path = tmp_data_dir_path / f"{SECTOR_NAME}_ch4_emi_flux.tif"

composting_dir = ghgi_data_dir_path / "composting"
sector_data_dir_path = data_dir_path / "sector"

# inventory workbook path
state_file_path = composting_dir / "State_Composting_1990-2021.xlsx"

# reference data paths
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"

# proxy input data paths

# input dataset 1: excess food opportunities data
excess_food_op_path = sector_data_dir_path / "CompostingFacilities.xlsx"
# input 2: frs facilities where NAICS code is for composting
frs_naics_path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV"
frs_facility_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV"
# input 3: biocycle facility locations pulled from v2
biocycle_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/"
    "Gridded CH4 Inventory - Task 2/ghgi_v3_working/GEPA_Source_Code/GEPA_Composting/"
    "InputData/biocycle_locs_clean.csv"
)
# input 4: ad council data from their kml file
comp_council_path = sector_data_dir_path / "STA Certified Compost Participants Map.kml"

# %%
# read in the state shapefile to spatial join with facilities, assigning them to
# states for allocation of emissions
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
state_gdf

# %%

# read in emission inventory
EPA_emissions = (
    pd.read_excel(
        state_file_path,
        sheet_name="InvDB",
        skiprows=15,
        nrows=115,
        usecols="A:AO",
    )
    # name column names lower
    .rename(columns=lambda x: str(x).lower())
    # drop columns we don't need
    .drop(
        columns=[
            "sector",
            "source",
            "subsource",
            "fuel",
            "subref",
            "2nd ref",
            "exclude",
        ]
    )
    # get just methane emissions
    .query("ghg == 'CH4'")
    # remove that column
    .drop(columns="ghg")
    # set the index to state
    .rename(columns={"state": "state_code"})
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
    .query("state_code.isin(@state_gdf['state_code'])")
)
EPA_emissions
# %%
# NOTE: there is a state that is significantly higher than the others here. CA?
sns.relplot(
    kind="line",
    data=EPA_emissions,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

# NOTE: Just checking
display(
    EPA_emissions.groupby("state_code")["ch4_kt"].mean().sort_values().reset_index()
)
# %%
# STEP 2: read in input data sets to format for proxy
"""
We read in 4 compost facility location datasets, aggregate them all together, and then
remove duplicate facilities based on their location by a tolerance in meters
If two facilities fall within the tolerance, only 1 location will be kept.
"""
excess_food_df = (
    pd.read_excel(
        excess_food_op_path,
        sheet_name="Data",
        usecols=["Name", "Latitude", "Longitude"],
    ).rename(columns=str.lower)
    # .rename(columns={"state": "state_code"})
    # .query("state_code.isin(@state_info_df['state_code'])")
    .assign(formatted_fac_name=lambda df: name_formatter(df["name"]), source="ex_food")
)
excess_food_df

# get the composting facilities from the FRS database based on the NAICS code.
# I use duckdb for a bit of performance in these very large tables over pandas.
frs_composting_fac_df = (
    duckdb.execute(
        (
            "SELECT frs_main.primary_name as name, frs_main.latitude83 as latitude, frs_main.longitude83 as longitude "
            f"FROM (SELECT registry_id, primary_name, latitude83, longitude83 FROM '{frs_facility_path}') as frs_main "
            f"JOIN (SELECT registry_id, naics_code FROM '{frs_naics_path}') AS frs_naics "
            "ON frs_main.registry_id = frs_naics.registry_id "
            f"WHERE naics_code == {COMPOSTING_FRS_NAICS_CODE}"
        )
    )
    .df()
    .assign(formatted_fac_name=lambda df: name_formatter(df["name"]), source="frs")
)
frs_composting_fac_df

biocycle_df = (
    pd.read_csv(biocycle_path)
    .rename(columns={"lat": "latitude", "lon": "longitude"})
    .assign(source="biocycle")
)
biocycle_df

# google earth exports 3d points, covert them to 2d to avoid issues.
_drop_z = lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2))

comp_council_gdf = (
    gpd.read_file(comp_council_path, driver="KML")
    .rename(columns=str.lower)
    .drop(columns="description")
    .assign(
        formatted_fac_name=lambda df: name_formatter(df["name"]),
        geometry=lambda df: df.geometry.transform(_drop_z),
        source="comp_council",
    )
)
comp_council_gdf

# %%
# there is a two step process to get all facilities in one dataframe: 1) put together
# the facilities that have lat/lon columns and make them geodataframes, 2) concat these
# with the facilities data that are already geodataframes
facility_concat_df = pd.concat(
    [frs_composting_fac_df, biocycle_df, excess_food_df]
).sort_values("formatted_fac_name")
facility_concat_gdf = gpd.GeoDataFrame(
    facility_concat_df.drop(columns=["latitude", "longitude"]),
    geometry=gpd.points_from_xy(
        facility_concat_df["longitude"], facility_concat_df["latitude"]
    ),
    crs=4326,
)
facility_concat_2_gdf = pd.concat([comp_council_gdf, facility_concat_gdf])
facility_concat_2_gdf = facility_concat_2_gdf[
    facility_concat_2_gdf.is_valid & ~facility_concat_2_gdf.is_empty
]
facility_concat_2_gdf.normalize()
facility_concat_2_gdf
# %%
# Remove duplicate facilities based on our spatial tolerance
# if two or more facilities fall within the tolerance, only 1 location is kept, the
# resulting centroid will be at the intersection of their buffers.
unique_facility_gdf = (
    facility_concat_2_gdf.to_crs("ESRI:102003")
    .assign(geometry=lambda df: df.buffer(DUPLICATION_TOLERANCE_M))
    .dissolve()
    .explode()
    .centroid.to_frame()
    .to_crs(4326)
    .sjoin(state_gdf[["geometry", "state_code"]])
    .drop(columns="index_right")
    .rename_geometry("geometry")
)
display(unique_facility_gdf.shape)
# NOTE: DC has no facilities listed, but has 1 year (2012) with reported emissions.
# So I proposed assigning the centroid of DC to our facilities list, to account for
# that one instance of not having data.
unique_facility_gdf = pd.concat(
    [
        unique_facility_gdf,
        state_gdf[state_gdf["state_code"] == "DC"][["state_code", "geometry"]].assign(
            geometry=lambda df: df.centroid
        ),
    ]
)
# %%
ax = unique_facility_gdf.plot("state_code", cmap="Set2", figsize=(10, 10))
state_gdf.boundary.plot(ax=ax, color="xkcd:slate")

# %%
facility_result_list = []
for (state, year), data in EPA_emissions.groupby(["state_code", "year"]):
    state_facilities = unique_facility_gdf[
        unique_facility_gdf["state_code"] == state
    ].copy()
    # print(state, year, state_facilities.shape)
    state_year_emissions = data["ch4_kt"].iat[0]
    if state_year_emissions == 0:
        continue
    if state_facilities.shape[0] < 1:
        print(state, year, "HAS NO FACILITIES")
        continue
    state_facilities["allocated_ch4_kt"] = (
        state_year_emissions / state_facilities.shape[0]
    )
    state_facilities = state_facilities.assign(year=year)
    facility_result_list.append(state_facilities)

proxy_data = pd.concat(facility_result_list).reset_index(drop=True)
# proxy_data.to_feather(tmp_data_dir_path / "TMP_composting_proxy.feather")
proxy_data
# %%

allocated_by_state_year = (
    proxy_data.groupby(["state_code", "year"])["allocated_ch4_kt"].sum().to_frame()
)

sns.relplot(
    kind="line",
    data=allocated_by_state_year,
    x="year",
    y="allocated_ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

sns.relplot(
    kind="line",
    data=EPA_emissions,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)
# %%
print("checking the allocation of emissions to proxy by state and year.")
check_sums = allocated_by_state_year.join(
    EPA_emissions.set_index(["state_code", "year"])
).assign(sums_equal=lambda df: np.isclose(df["allocated_ch4_kt"], df["ch4_kt"]))
display(check_sums)
all_equal = check_sums["sums_equal"].all()

print(f"do all allocated emission by state and year equal (isclose): {all_equal}")
if not all_equal:
    print("if not, these ones below DO NOT equal")
    display(check_sums[~check_sums["sums_equal"]])
# %%
area_matrix = load_area_matrix()
# %%
# STEP 5: RASTERIZE THE CH4 KT AND FLUX
# e.g., calculate fluxes and place the facility-level emissions on the CONUS grid

# for each year, grid the adjusted emissions data in kt and do conversion for flux.
ch4_kt_result_rasters = {}
ch4_flux_result_rasters = {}


for year, data in proxy_data.groupby("year"):
    data = data[(data.is_valid) & (~data.is_empty)]
    # same results as summing the month days
    # if calendar.isleap(year):
    #     year_days = 366
    # else:
    #     year_days = 365
    month_days = [calendar.monthrange(year, x)[1] for x in range(1, 13)]
    year_days = np.sum(month_days)

    # TODO: check that when multiple points fall into the same cell, their values
    # are added together.
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
# %%
print("checking gridded result values by year.")
check_sum_dict = {}
for year, arr in ch4_kt_result_rasters.items():
    gridded_sum = arr.sum()
    check_sum_dict[year] = gridded_sum

gridded_year_sums_df = (
    pd.DataFrame()
    .from_dict(check_sum_dict, orient="index")
    .rename(columns={0: "gridded_sum"})
)

emissions_by_year_check = (
    EPA_emissions.groupby("year")["ch4_kt"]
    .sum()
    .to_frame()
    .join(gridded_year_sums_df)
    .assign(isclose=lambda df: np.isclose(df["ch4_kt"], df["gridded_sum"]))
)
all_equal = emissions_by_year_check["isclose"].all()

print(f"do all gridded emission by year equal (isclose): {all_equal}")
if not all_equal:
    print("if not, these ones below DO NOT equal")
    display(emissions_by_year_check[~emissions_by_year_check["isclose"]])
# %%
write_tif_output(ch4_kt_result_rasters, ch4_kt_dst_path)
write_tif_output(ch4_flux_result_rasters, ch4_flux_dst_path)
# %%
# TODO: write netcdf outputs
# TODO: create visuals/maps
# %%
