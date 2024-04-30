# %%
# %load_ext autoreload
# %autoreload 2
# %%
import calendar
import warnings
from pathlib import Path

import osgeo  # noqa
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import seaborn as sns
from IPython.display import display
# from pytask import Product, task
from rasterio.features import rasterize

from gch4i.config import (ghgi_data_dir_path, max_year, min_year,
                          tmp_data_dir_path)
from gch4i.gridding import ARR_SHAPE, GEPA_PROFILE
from gch4i.utils import calc_conversion_factor, load_area_matrix, write_outputs

# %%

# @mark.persist
# @task
# def task_sector_industry_ferroalloy():
#     pass


# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
INDUSTRY_NAME = "2C2_industry_ferroalloy"

# NOTE: We expect both the summary emissions and the proxy to come from the same file
# for this sector.
EPA_inputfile = Path(ghgi_data_dir_path / "State_Ferroalloys_1990-2021.xlsx")

# NOTE: this file is temporarily used to get the location of facilities. We expect the
# inventory team to deliver the final workbook with facility locations in it.
EPA_ghgrp_ferrofacility_inputfile = (
    ghgi_data_dir_path / "SubpartK_Ferroalloy_Facilities.csv"
)

# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{INDUSTRY_NAME}_ch4_kt_per_year.tif"
ch4_flux_dst_path = tmp_data_dir_path / f"{INDUSTRY_NAME}_ch4_emi_flux.tif"

# NOTE: I think it makes sense to still load the area matrix in here so it persists
# in memory and we pass it to the flux calculation function when needed.
area_matrix = load_area_matrix()

# %%

# NOTE: looking at rework of the proxy mapping files into an aggregate flat file
# that would then be formed into a proxy dictionary to retain the existing approach
# but allow for interrogation of the objects when needed.
proxy_mapping_dir = Path(
    "C:/Users/nkruskamp/Research Triangle Institute/EPA Gridded Methane - Task 2/data"
)
ghgi_map_path = proxy_mapping_dir / "all_ghgi_mappings.csv"
proxy_map_path = proxy_mapping_dir / "all_proxy_mappings.csv"
ghgi_map_df = pd.read_csv(ghgi_map_path)
proxy_map_df = pd.read_csv(proxy_map_path)

proxy_map_df.query("GHGI_Emi_Group == 'Emi_Ferro'").merge(
    ghgi_map_df.query("GHGI_Emi_Group == 'Emi_Ferro'"), on="GHGI_Emi_Group"
)

ferro_map_data_dict = {}
ferro_map_data_dict["Emi_Ferro"] = {
    "Map_Ferro": {
        "facility_data_file": EPA_inputfile,
        "sheet_name": "Ferroalloy Calculations",
    }
}
# #load GHGI Mapping Groups
# names = pd.read_excel(Ind_Mapping_inputfile, sheet_name = "GHGI Map - Ind", usecols = "A:B",skiprows = 1, header = 0)
# colnames = names.columns.values
# ghgi_ind_map = pd.read_excel(Ind_Mapping_inputfile, sheet_name = "GHGI Map - Ind", usecols = "A:B", skiprows = 1, names = colnames)
# #drop rows with no data, remove the parentheses and ""
# ghgi_ind_map = ghgi_ind_map[ghgi_ind_map['GHGI_Emi_Group'] != 'na']
# ghgi_ind_map = ghgi_ind_map[ghgi_ind_map['GHGI_Emi_Group'].notna()]
# ghgi_ind_map['GHGI_Source']= ghgi_ind_map['GHGI_Source'].str.replace(r"\(","")
# ghgi_ind_map['GHGI_Source']= ghgi_ind_map['GHGI_Source'].str.replace(r"\)","")
# ghgi_ind_map.reset_index(inplace=True, drop=True)
# display(ghgi_ind_map)

# #load emission group - proxy map
# names = pd.read_excel(Ind_Mapping_inputfile, sheet_name = "Proxy Map - Ind", usecols = "A:D",skiprows = 1, header = 0)
# colnames = names.columns.values
# proxy_ind_map = pd.read_excel(Ind_Mapping_inputfile, sheet_name = "Proxy Map - Ind", usecols = "A:D", skiprows = 1, names = colnames)
# display((proxy_ind_map))

# #create empty proxy and emission group arrays (add months for proxy variables that have monthly data)
# for igroup in np.arange(0,len(proxy_ind_map)):
#     vars()[proxy_ind_map.loc[igroup,'Proxy_Group']] = np.zeros([len(Lat_01),len(Lon_01),num_years])
#     vars()[proxy_ind_map.loc[igroup,'Proxy_Group']+'_nongrid'] = np.zeros([num_years])
#     vars()[proxy_ind_map.loc[igroup,'GHGI_Emi_Group']] = np.zeros([num_years])


# %%


# state_gdf = gpd.read_file(
#     "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
# )


# %%
# STEP 1: GET STATE EMISSIONS DATA BY YEAR

# read in the ch4_kt values for each state
# This pulls from the full table, but ends up the same as the CH4 kt table found on
# sheet "Ferroallow_State_Disagg_calcs"
# EPA_ferro_emissions = (
#     # read in the data
#     pd.read_excel(
#         EPA_inputfile,
#         sheet_name="State Summary",
#         skiprows=124,
#         nrows=57,
#         usecols="B:AJ",
#     )
#     # name column names lower
#     .rename(columns=lambda x: str(x).lower())
#     # drop columns we don't need
#     .drop(columns=["source", "ghg"])
#     # remove the national total column
#     .query("state != 'National'")
#     # set the index as the state
#     .set_index("state")
#     # drop states that are all na / have no data
#     .dropna(how="all")
#     # reset the index
#     .reset_index()
#     # make the table long
#     .melt(id_vars="state", var_name="year", value_name="ch4_kt")
#     .astype({"year": int})
#     .query("year.between(@min_year, @max_year)")
# )
EPA_ferro_emissions = (
    # read in the data
    pd.read_excel(
        EPA_inputfile,
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
    .set_index("state")
    # covert "NO" string to numeric (will become np.nan)
    .apply(pd.to_numeric, errors="coerce")
    # drop states that have all nan values
    .dropna(how="all")
    # reset the index state back to a column
    .reset_index()
    # make the table long by state/year
    .melt(id_vars="state", var_name="year", value_name="ch4_tg")
    .assign(ch4_kt=lambda df: df["ch4_tg"] * 1000)
    .drop(columns=["ch4_tg"])
    # make the columns types correcet
    .astype({"year": int, "ch4_kt": float})
    .fillna({"ch4_kt": 0})
    # get only the years we need
    .query("year.between(@min_year, @max_year)")
)
EPA_ferro_emissions.head()
# %%
display(EPA_ferro_emissions["state"].value_counts())
display(EPA_ferro_emissions["year"].min(), EPA_ferro_emissions["year"].max())

# a quick plot to verify the values
sns.relplot(
    kind="line",
    data=EPA_ferro_emissions,
    x="year",
    y="ch4_kt",
    hue="state",
    # legend=False,
)
# %%


# STEP 2: GET PROXY DATA
# TODO: explore if it makes sense to write a function / task for every proxy.
# there are about 65 unique ones.
def task_map_ferro_proxy():
    pass


# The facilities have multiple reporting units for each year. This will read in the
# facilities data and compute the facility level sum of ch4_kt emissions for each
# year. This pulls from the raw table but ends in the same form as the table on sheet
# "GHGRP_kt_Totals"

# read in the SUMMARY facilities emissions data
ferro_facilities_df = (
    pd.read_excel(
        EPA_inputfile,
        sheet_name="Ferroalloy Calculations",
        skiprows=72,
        nrows=12,
        usecols="A:AI",
    )
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"facility": "facility_name", "state": "fac_state"})
    .drop(columns=["year opened"])
    .melt(id_vars=["facility_name", "fac_state"], var_name="year", value_name="ch4_kt")
    .astype({"year": int})
    .assign(
        formatted_fac_name=lambda df: df["facility_name"]
        .str.replace("  ", " ")
        .replace("[^a-zA-Z0-9 -]", "", regex=True)  # .str.casefold()
    )
    .query("year.between(@min_year, @max_year)")
)

# all_facilities = (
#     pd.read_excel(
#         EPA_inputfile,
#         sheet_name="noncbi_k_gasinfo",
#         skiprows=1,
#         # nrows=12,
#         # usecols="A:AI",
#     )
#     .rename(columns=lambda x: str(x).lower())
#     .query("ghgasname == 'Methane'")
#     .drop(columns=["submission_id", "ghgasname", "otherghgasname"])
#     .rename(columns={"reporting_year": "year", "ghgasquantity": "ch4_mt"})
#     # .melt(id_vars=["facility_name", "fac_state"], var_name="year", value_name="ch4_kt")
#     .astype({"year": int})
#     .assign(
#         formatted_fac_name=lambda df: df["facility_name"]
#         .str.replace("  ", " ")
#         .replace("[^a-zA-Z0-9 -]", "", regex=True)  # .str.casefold()
#     )
#     .query("year.between(@min_year, @max_year)")
# )

# # read in the combined USGS / EPA facilities data
# all_facilities = (
#     pd.read_excel(
#         EPA_inputfile,
#         sheet_name="GHGRP_USGS_Facilities",
#         skiprows=2,
#         # nrows=57,
#         # usecols="B:AJ",
#     )
#     .rename(columns=lambda x: str(x).lower())
#     .rename(columns={"reporting_year": "year", "ch4 (kt)": "ch4_kt"})
#     .loc[:, ["facility_id", "facility_name", "year", "ch4_kt"]]
#     .groupby(["facility_id", "facility_name", "year"])["ch4_kt"]
#     .sum()
#     .reset_index()
#     .assign(
#         formatted_fac_name=lambda df: df["facility_name"]
#         .str.replace("  ", " ")
#         .replace("[^a-zA-Z0-9 -]", "", regex=True)
#     )
#     .query("year.between(@min_year, @max_year)")
# )

# NOTE: again this is temporary to get locations to work with for now. When final
# workbook is delivered this will be updated to get locations.
ferro_subk_info_df = pd.read_csv(EPA_ghgrp_ferrofacility_inputfile).rename(
    columns=lambda x: str(x).split(".")[-1].lower()
)
ferro_subk_info_gdf = (
    gpd.GeoDataFrame(
        ferro_subk_info_df,
        geometry=gpd.points_from_xy(
            ferro_subk_info_df["longitude"],
            ferro_subk_info_df["latitude"],
            crs=4326,
        ),
    )
    .assign(
        formatted_fac_name=lambda df: df["facility_name"]
        .str.replace("  ", " ")
        .replace("[^a-zA-Z0-9 -]", "", regex=True)  # .str.casefold()
    )
    .loc[:, ["formatted_fac_name", "state", "zip", "geometry"]]
    .drop_duplicates(["formatted_fac_name", "state", "zip"])
)

# merge the location data with the facility emissions data
ferro_facilities_gdf = ferro_subk_info_gdf.merge(
    ferro_facilities_df,
    on=["formatted_fac_name"],
    how="right",
)

ferro_facilities_gdf.head()
# %%
# quick look at the data.
sns.lineplot(
    ferro_facilities_gdf, x="year", y="ch4_kt", hue="facility_name", legend=False
)
# %%
# EPA_ghgrp_ferro_inputfile = ghgi_path / "SubpartK_Ferroalloy.csv"
# ferro_subk_emi_df = (
#     pd.read_csv(EPA_ghgrp_ferro_inputfile)
#     .rename(columns=lambda x: str(x).split(".")[-1].lower())
#     .rename(columns={"reporting_year": "year"})
#     .query("ghg_name == 'Methane'")
#     .drop(columns="ghg_name")
# )

# v2_facility_info_df = (
#     ferro_subk_emi_df.set_index(["facility_id", "year"])
#     .join(ferro_subk_info_df.set_index(["facility_id", "year"]))
#     .reset_index()
# )
# %%
# make sure the merge gave us the number of results we expected.
if not (ferro_facilities_gdf.shape[0] == ferro_facilities_df.shape[0]):
    print("WARNING the merge shape does not match the original data")
# %%

# save a shapefile of the v3 ferro facilities for reference
fac_locations = ferro_facilities_gdf.dissolve("formatted_fac_name")
fac_locations[fac_locations.is_valid].loc[:, ["geometry"]].to_file(
    tmp_data_dir_path / "v3_ferro_facilities.shp.zip", driver="ESRI Shapefile"
)


# %%

# some checks of the data
# how many na values are there
display(ferro_facilities_gdf.isna().sum())
# how many missing locations are there by year
display(ferro_facilities_gdf[ferro_facilities_gdf["zip"].isna()]["year"].value_counts())
# how many missing locations are there by facility name
display(
    ferro_facilities_gdf[ferro_facilities_gdf["zip"].isna()][
        "formatted_fac_name"
    ].value_counts()
)
# a plot of the timeseries of emission by facility
sns.lineplot(
    data=ferro_facilities_gdf, x="year", y="ch4_kt", hue="facility_name", legend=False
)

# %%

# NOTE: checking if the facility values equal the summary values.
# # check the state/year sums against state summary data
# for name, data in EPA_ferro_emissions.groupby(["state", "year"]):
#     emi_sum = data["ch4_kt"].sum()
#     # fac_sum = data["ch4_kt"].sum()

#     state_facs = ferro_facilities_gdf[
#         (ferro_facilities_gdf["state"] == name[0]) & (ferro_facilities_gdf["year"] == name[1])
#     ]

#     fac_sum = state_facs["ch4_kt"].sum()
#     if state_facs.shape[0] == 0:
#         print(name, "no facility data")
#     else:
#         print(name, np.isclose(emi_sum, fac_sum))

# # check just the year sums against the state summary data
# for year, data in EPA_ferro_emissions.groupby("year"):
#     emi_sum = data["ch4_kt"].sum()
#     fac_sum = ferro_facilities_gdf.query("year == @year")["ch4_kt"].sum()
#     print(year, np.isclose(fac_sum, emi_sum))

# %%

# STEP 3: ALLOCATION STATE / YEAR EMISSIONS BY PROXY FRACTION
# ALLOCATE the relative facility emissions data by state/year to the state summary
# emissions

# This does the allocation for us in a function by state and year.


def state_year_allocation_emissions(fac_emissions, inventory_df):

    # fac_emissions are the reported emission for the facilities located in that state
    # and year. It can be one or more facilities. Inventory_df is the summary
    # emissions table data we query based on the state and year.

    # get the target state and year
    state, year = fac_emissions.name
    # get the summary emissions for that state and year. It will be a single value.
    emi_sum = inventory_df[
        (inventory_df["state"] == state) & (inventory_df["year"] == year)
    ]["ch4_kt"].iat[0]

    # allocate the summary emissions to the individual facilities based on their
    # proportion of reported emissions.
    allocated_fac_emissions = ((fac_emissions / fac_emissions.sum()) * emi_sum).fillna(
        0
    )
    return allocated_fac_emissions


# we create a new column that assigns the allocated summary emissions to each facility
# based on its proportion of emission to the facility totals for that state and year.
# so for each state and year in the summary emissions we apply the function.
ferro_facilities_gdf["allocated_ch4_kt"] = ferro_facilities_gdf.groupby(
    ["state", "year"]
)["ch4_kt"].transform(state_year_allocation_emissions, inventory_df=EPA_ferro_emissions)
# %%
# We now check that the sum of facility emissions equals the summary emissions by state
# and year. The resulting sum_check table shows you where the emissions data DO NOT
# equal and need more investigation.
# NOTE: currently we are missing facilities in states, so we also check below that the
# states that are missing emissions are the ones that are missing facilities.
sum_check = (
    ferro_facilities_gdf.groupby(["state", "year"])["allocated_ch4_kt"]
    .sum()
    .reset_index()
    .merge(EPA_ferro_emissions, on=["state", "year"], how="outer")
    .assign(
        check_diff=lambda df: df.apply(
            lambda x: np.isclose(x["allocated_ch4_kt"], x["ch4_kt"]), axis=1
        )
    )
)
display(sum_check[~sum_check["check_diff"]])

# NOTE: For now, facilities data are not final / missing. We don't have facilities in
# all the state summaries that are reporting, and we may be missing facilities even
# within states that are represented. If these lists match, we have a good idea of
# what is missing currently due to the preliminary data.
print(
    (
        "states with no facilities in them: "
        f"{EPA_ferro_emissions[~EPA_ferro_emissions['state'].isin(ferro_facilities_gdf['state'])]['state'].unique()}"
    )
)
print(
    (
        "states with unaccounted emissions: "
        f"{sum_check[~sum_check['check_diff']]['state'].unique()}"
    )
)

# %%

# STEP 4: RASTERIZE THE CH4 KT AND FLUX

# for each year, grid the adjusted emissions data in kt and do conversion for flux.
ch4_kt_result_rasters = {}
ch4_flux_result_rasters = {}

# NOTE: this warning filter is because we currently have facilities with missing
# geometries.
# TODO: remove this filter when full locations data are available.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for year, data in ferro_facilities_gdf.groupby("year"):

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
# STEP 4.1: SAVE THE FILES

# check the sums all together now...
# TODO: report QC metrics for both kt and flux values
for year, raster in ch4_kt_result_rasters.items():
    raster_sum = raster.sum()
    fac_sum = ferro_facilities_gdf.query("year == @year")["allocated_ch4_kt"].sum()
    emi_sum = EPA_ferro_emissions.query("year == @year")["ch4_kt"].sum()
    missing_sum = sum_check[(~sum_check["check_diff"]) & (sum_check["year"] == year)][
        "ch4_kt"
    ].sum()

    print(year)
    print(
        "does the raster sum equal the facility sum: "
        f"{np.isclose(raster_sum, fac_sum)}"
    )
    print(
        "does the raster sum equal the national total: "
        f"{np.isclose(raster_sum, emi_sum)}"
    )
    # this shows we are consistent with our missing emissions where the states with no
    # facilities make up the difference, as we would expect.
    print(
        "do the states with no facilities equal the missing amount: "
        f"{np.isclose((emi_sum - raster_sum), missing_sum)}"
    )
    print()

# %%
write_outputs(ch4_kt_result_rasters, ch4_kt_dst_path)
write_outputs(ch4_flux_result_rasters, ch4_flux_dst_path)
# %%
