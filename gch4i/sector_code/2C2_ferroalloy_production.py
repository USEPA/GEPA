"""
Name:               2C2_ferroalloy_production.py
Authors Name:       N. Kruskamp, H. Lohman (RTI International), Erin McDuffie (EPA/OAP)
Date Last Modified: 06/10/2024
Purpose:            Spatially allocates methane emissions for source category 2C2
                    Ferroalloy production
Input Files:
                    - 
Output Files:
                    - {FULL_NAME}_ch4_kt_per_year.tif
                    - {FULL_NAME}_ch4_emi_flux.tif
Notes:
TODO: get latest inventory data
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------

# for testing/development
# %load_ext autoreload
# %autoreload 2

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
)

# from pytask import Product, task


gpd.options.io_engine = "pyogrio"

# TODO: move to emis file
def get_ferro_inventory_data(input_path):
    """read in the ghgi_ch4_kt values for each state"""
    
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
        # drop columns we don't need
        .drop(
            columns=[
                "Data Type",
                "Sector",
                "Subsector",
                "Category",
                "Subcategory1",
                "Subcategory2",
                "Subcategory3",
                "Subcategory4",
                "Subcategory5",
                "Carbon Pool",
                "Fuel1",
                "Fuel2",
                # "GeoRef",
                "Exclude",
                "CRT Code",
                "ID",
                "Sensitive (Y or N)",
                "Units",
                # "GHG",
                "GWP",
            ]
        )
        .rename(columns=lambda x: str(x).lower())
        # get just methane emissions
        .query("ghg == 'CH4'")
        # remove that column
        .drop(columns="ghg")
        # set the index to state
        .rename(columns={"georef": "state_code"})
        .query("state_code != 'National'")
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
    )
    return emi_df


def get_ferro_proxy_data(EPA_inventory_path, frs_path, subpart_k_url, state_gdf):
    # The facilities have multiple reporting units for each year. This will read in the
    # facilities data and compute the facility level sum of emissions for each
    # year. This pulls from the raw table but ends in the same form as the table on sheet
    # "GHGRP_kt_Totals"

    # read in the SUMMARY facilities emissions data
    ferro_facilities_df = (
        pd.read_excel(
            EPA_inventory_path,
            sheet_name="Ferroalloy Calculations",
            skiprows=72,
            nrows=12,
            usecols="A:AJ",
        )
        .rename(columns=lambda x: str(x).lower())
        .rename(columns={"facility": "facility_name", "state": "state_name"})
        .drop(columns=["year opened"])
        .melt(
            id_vars=["facility_name", "state_name"],
            var_name="year",
            value_name="est_ch4",
        )
        .astype({"year": int})
        .assign(formatted_fac_name=lambda df: name_formatter(df["facility_name"]))
        .merge(state_gdf[["state_code", "state_name"]], on="state_name")
        .query("year.between(@min_year, @max_year)")
    )

    # STEP 3.1: Get and format FRS and Subpart K facility locations data
    # this will read in both datasets, format the columns for the facility name, 2 letter
    # state code, latitude and longitude. Then merge the two tables together, and create
    # a formatted facility name that eases matching of our needed facilities to the location
    # data

    frs_raw = duckdb.execute(
        (
            "SELECT primary_name, state_code, latitude83, longitude83 "
            f"FROM '{frs_path}' "
        )
    ).df()

    subpart_k_raw = pd.read_csv(subpart_k_url)

    frs_df = (
        frs_raw.rename(columns=str.lower)
        .drop_duplicates(subset=["primary_name", "state_code"])
        .rename(columns={"primary_name": "facility_name"})
        .rename(columns=lambda x: x.strip("83"))
        .dropna(subset=["latitude", "longitude"])
    )

    subpart_k_df = (
        subpart_k_raw.drop_duplicates(subset=["facility_name", "state"])
        .rename(columns={"state": "state_code"})
        .loc[
            :,
            ["facility_name", "state_code", "latitude", "longitude"],
        ]
        .dropna(subset=["latitude", "longitude"])
    )

    # merge the two datasets together, format the facility name, drop any duplicates.
    facility_locations = (
        pd.concat([subpart_k_df, frs_df], ignore_index=True)
        .assign(formatted_fac_name=lambda df: name_formatter(df["facility_name"]))
        .drop_duplicates(subset=["formatted_fac_name", "state_code"])
    )

    # create a table of unique facility names to match against the facility locations data.
    unique_facilities = (
        ferro_facilities_df[["formatted_fac_name", "state_code"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # try to match facilities based on the formatted name and state code
    matching_facilities = unique_facilities.merge(
        facility_locations, on=["formatted_fac_name", "state_code"], how="left"
    )

    # NOTE: We are missing 4 facilities. Manually, we checked the 4 missing facilities
    # for partial name matches in the subpart/frs facility data. We found 3 facilities.
    # 1 was still missing. Using partial matching of name and state, we assign the coords
    # to our records.

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "bear metallurgical co",
        ["facility_name", "latitude", "longitude"],
    ] = (
        facility_locations[
            facility_locations["formatted_fac_name"].str.contains("bear metal")
            & facility_locations["state_code"].eq("PA")
        ]
        .loc[:, ["facility_name", "latitude", "longitude"]]
        .values
    )

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "stratcor inc",
        ["facility_name", "latitude", "longitude"],
    ] = (
        facility_locations[
            facility_locations["formatted_fac_name"].str.contains("stratcor")
            & facility_locations["state_code"].eq("AR")
        ]
        .loc[:, ["facility_name", "latitude", "longitude"]]
        .values
    )

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "metallurg vanadium corp",
        ["facility_name", "latitude", "longitude"],
    ] = (
        facility_locations[
            facility_locations["formatted_fac_name"].str.contains("vanadium inc")
            & facility_locations["state_code"].eq("OH")
        ]
        .loc[:, ["facility_name", "latitude", "longitude"]]
        .values
    )

    # NOTE: The final facility, thompson creek, has no name match in the facilities data,
    # so we use the city and state provided in the inventory data to get its location.
    # https://en.wikipedia.org/wiki/Thompson_Creek_Metals.
    # thompson creek metals co inc was acquired in 2016 by Centerra Gold.
    # A search for Centerra in PA turns up 2 records in the combine subpart K and FRS
    # dataset. But Centerra Co-op is an agricultural firm, not a ferroalloy facility.
    # https://www.centerracoop.com/locations
    # display(
    #     facility_locations[
    #         facility_locations["formatted_fac_name"].str.contains("centerra")
    #         & facility_locations["state_code"].eq("PA")
    #     ]
    # )

    fac_locations = pd.read_excel(
        inventory_workbook_path,
        sheet_name="USGS_2008_Facilities",
        skiprows=4,
        nrows=12,
        usecols="A:C",
    ).drop(columns="Unnamed: 1")
    t_creek_city_state = fac_locations.loc[
        fac_locations["Company"] == "Thompson Creek Metals Co.", "Plant location"
    ].values[0]

    geolocator = Nominatim(user_agent="RTI testing")
    t_creek_loc = geolocator.geocode(t_creek_city_state)

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "thompson creek metals co inc",
        ["latitude", "longitude"],
    ] = [t_creek_loc.latitude, t_creek_loc.longitude]

    # we now have all facilities with addresses.
    ferro_facilities_gdf = ferro_facilities_df.merge(
        matching_facilities.drop(columns="facility_name"),
        on=["formatted_fac_name", "state_code"],
        how="left",
    )
    ferro_facilities_gdf = gpd.GeoDataFrame(
        ferro_facilities_gdf.drop(columns=["latitude", "longitude"]),
        geometry=gpd.points_from_xy(
            ferro_facilities_gdf["longitude"],
            ferro_facilities_gdf["latitude"],
            crs=4326,
        ),
    )

    # make sure the merge gave us the number of results we expected.
    if not (ferro_facilities_gdf.shape[0] == ferro_facilities_df.shape[0]):
        print("WARNING the merge shape does not match the original data")
    ferro_facilities_gdf
    # # save a shapefile of the v3 ferro facilities for reference
    # fac_locations = ferro_facilities_gdf.dissolve("formatted_fac_name")
    # fac_locations[fac_locations.is_valid].loc[:, ["geometry"]].to_file(
    #     tmp_data_dir_path / "v3_ferro_facilities.shp.zip", driver="ESRI Shapefile"
    # )
    return ferro_facilities_gdf


# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = "2C2"
SECTOR_NAME = "industry"
SOURCE_NAME = "ferroalloy production"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

VERSION = "draft"

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID} - {VERSION}"
)

# PATHS

# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# INVENTORY INPUT FILE
# XXX: prelim 2024 report data.
inventory_workbook_path = ghgi_data_dir_path / "State_Ferroalloys_1990-2022.xlsx"

# PROXY INPUT FILES
# state vector refence with state_code
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
# input 1: ferro facilities. NOTE: we the list of ferro facilities from the workbook.
# input 2: subpart data are retrieve via the API
subart_k_api_url = (
    "https://data.epa.gov/efservice/k_subpart_level_information/"
    "pub_dim_facility/ghg_name/=/Methane/CSV"
)
# input 3: frs facilities
frs_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV"

# %% STEP 1. Load GHGI-Proxy Mapping Files ---------------------------------------------
# NOTE: looking at rework of the proxy mapping files into an aggregate flat file
# that would then be formed into a proxy dictionary to retain the existing approach
# but allow for interrogation of the objects when needed.
# EEM: updated to v3 path
# proxy_mapping_dir = Path(
#    "C:/Users/nkruskamp/Research Triangle Institute/EPA Gridded Methane - Task 2/data"
# )
# ghgi_map_path = proxy_mapping_dir / "all_ghgi_mappings.csv"
# proxy_map_path = proxy_mapping_dir / "all_proxy_mappings.csv"
ghgi_map_path = V3_DATA_PATH / "all_ghgi_mappings.csv"
proxy_map_path = V3_DATA_PATH / "all_proxy_mappings.csv"
ghgi_map_df = pd.read_csv(ghgi_map_path)
proxy_map_df = pd.read_csv(proxy_map_path)

proxy_map_df.query("GHGI_Emi_Group == 'Emi_Ferro'").merge(
    ghgi_map_df.query("GHGI_Emi_Group == 'Emi_Ferro'"), on="GHGI_Emi_Group"
)

ferro_map_data_dict = {}
ferro_map_data_dict["Emi_Ferro"] = {
    "Map_Ferro": {
        "facility_data_file": inventory_workbook_path,
        "sheet_name": "Ferroalloy Calculations",
    }
}

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
# %%
EPA_state_emi_df = get_ferro_inventory_data(inventory_workbook_path)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_emi_df,
    x="year",
    y="ghgi_ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

# %% STEP 3: GET AND FORMAT PROXY DATA -------------------------------------------------
ferro_proxy_gdf = get_ferro_proxy_data(
    inventory_workbook_path, frs_path, subart_k_api_url, state_gdf
)

# %% STEP 3.1: QA/QC proxy data --------------------------------------------------------

# timeseries look at the data.
g = sns.lineplot(
    ferro_proxy_gdf,
    x="year",
    y="est_ch4",
    hue="facility_name",
    legend=True,
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 0.9))
sns.despine()
# some checks of the data
# how many na values are there
print("Number of NaN values:")
display(ferro_proxy_gdf.isna().sum())
# how many missing locations are there by year
print("Number of Facilities with Missing Locations Each Year")
display(ferro_proxy_gdf[ferro_proxy_gdf.is_empty]["year"].value_counts())
# how many missing locations are there by facility name
print("For Each Facility with Missing Data, How Many Missing Years")
display(ferro_proxy_gdf[ferro_proxy_gdf.is_empty]["formatted_fac_name"].value_counts())
# a plot of the timeseries of emission by facility
sns.lineplot(
    data=ferro_proxy_gdf, x="year", y="est_ch4", hue="facility_name", legend=False
)

# %% MAP PROXY DATA --------------------------------------------------------------------
ax = ferro_proxy_gdf.drop_duplicates("formatted_fac_name").plot(
    "formatted_fac_name", categorical=True, cmap="Set2", figsize=(10, 10)
)
state_gdf.boundary.plot(ax=ax, color="xkcd:slate", lw=0.2, zorder=1)

# %% STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES ---------------------------
allocated_emis_gdf = allocate_emissions_to_proxy(
    ferro_proxy_gdf,
    EPA_state_emi_df,
    proxy_has_year=True,
    use_proportional=True,
    proportional_col_name="est_ch4",
)
allocated_emis_gdf

# %% STEP 4.1: QC PROXY ALLOCATED EMISSIONS BY STATE AND YEAR --------------------------
proxy_qc_result = QC_proxy_allocation(allocated_emis_gdf, EPA_state_emi_df)
proxy_qc_result

# %% STEP 5: RASTERIZE THE CH4 KT AND FLUX ---------------------------------------------
ch4_kt_result_rasters = grid_allocated_emissions(allocated_emis_gdf)
ch4_flux_result_rasters = calculate_flux(ch4_kt_result_rasters)


# %% STEP 5.1: QC GRIDDED EMISSIONS BY YEAR --------------------------------------------
# TODO: report QC metrics for flux values compared to V2: descriptive statistics
qc_kt_rasters = QC_emi_raster_sums(ch4_kt_result_rasters, EPA_state_emi_df)
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
