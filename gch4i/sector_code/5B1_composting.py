"""
Name:               5B1_composting.py
Authors Name:       N. Kruskamp, H. Lohman (RTI International), Erin McDuffie (EPA/OAP)
Date Last Modified: 06/10/2024
Purpose:            Spatially allocates methane emissions for source category 5B1
                    composting.
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
from IPython.display import display
from shapely import Point, wkb

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
)

# from pytask import Product, task


gpd.options.io_engine = "pyogrio"


def get_composting_inventory_data(input_path):
    emi_df = (
        pd.read_excel(
            input_path,
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
    return emi_df


def get_composting_proxy_data(
    excess_food_op_path,
    frs_facility_path,
    frs_naics_path,
    biocycle_path,
    comp_council_path,
    state_gdf,
):

    excess_food_df = (
        pd.read_excel(
            excess_food_op_path,
            sheet_name="Data",
            usecols=["Name", "Latitude", "Longitude"],
        ).rename(columns=str.lower)
        # .rename(columns={"state": "state_code"})
        # .query("state_code.isin(@state_info_df['state_code'])")
        .assign(
            formatted_fac_name=lambda df: name_formatter(df["name"]), source="ex_food"
        )
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

    # there is a two step process to get all facilities in one dataframe:
    # 1) put together the facilities that have lat/lon columns and make them
    # geodataframes,
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
    # 2) concat these with the facilities data that are already geodataframes.
    facility_concat_2_gdf = pd.concat([comp_council_gdf, facility_concat_gdf])
    facility_concat_2_gdf = facility_concat_2_gdf[
        facility_concat_2_gdf.is_valid & ~facility_concat_2_gdf.is_empty
    ]
    facility_concat_2_gdf.normalize()

    # Remove duplicate facilities based on our spatial tolerance
    # if two or more facilities fall within the tolerance, only 1 location is kept, the
    # resulting centroid will be at the intersection of their buffers.
    final_facility_gdf = (
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

    # NOTE: DC has no facilities listed, but has 1 year (2012) with reported emissions.
    # Erin email note on DC: My understanding is that (industrial) composting in DC has
    # been outsources to MD. Since the inventory is trying to capture the distribution
    # of industrial composting emissions, let’s manually add a single facility location
    # for DC. Let’s use the Fort Totten Waste Transfer Station (lat/lon: 38.947624,
    # -77.001213). We can make a note of this in the data and assumptions document and
    # this assumption can be re-visited in later iterations.
    final_facility_gdf = pd.concat(
        [
            final_facility_gdf,
            state_gdf[state_gdf["state_code"] == "DC"][
                ["state_code", "geometry"]
            ].assign(geometry=Point(-77.001213, 38.947624)),
        ]
    ).reset_index(drop=True)
    return final_facility_gdf


# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = "5B1"
SECTOR_NAME = "waste"
SOURCE_NAME = "composting"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
)

# PATHS
composting_dir = ghgi_data_dir_path / "composting"
sector_data_dir_path = V3_DATA_PATH / "sector"

# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# INVENTORY INPUT FILE
inventory_workbook_path = composting_dir / "State_Composting_1990-2021.xlsx"

# PROXY INPUT FILES
# state vector refence with state_code
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
# input 1: excess food opportunities data
excess_food_op_path = sector_data_dir_path / "CompostingFacilities.xlsx"
# input 2: frs facilities where NAICS code is for composting
frs_naics_path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV"
frs_facility_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV"
# input 3: biocycle facility locations pulled from v2
biocycle_path = sector_data_dir_path / "biocycle_locs_clean.csv"
# input 4: ad council data from their kml file
comp_council_path = sector_data_dir_path / "STA Certified Compost Participants Map.kml"

# the NAICS code pulled from the v2 notebook for facilities in the FRS data
COMPOSTING_FRS_NAICS_CODE = 562219
# the spatial tolerance for removing duplicate facility points
DUPLICATION_TOLERANCE_M = 250
# %% STEP 1. Load GHGI-Proxy Mapping Files ---------------------------------------------

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

EPA_state_emi_df = get_composting_inventory_data(inventory_workbook_path)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_emi_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

# %% STEP 3: GET AND FORMAT PROXY DATA -------------------------------------------------
composting_proxy_gdf = get_composting_proxy_data(
    excess_food_op_path,
    frs_facility_path,
    frs_naics_path,
    biocycle_path,
    comp_council_path,
    state_gdf,
)

# %% STEP 3.1: QA/QC proxy data --------------------------------------------------------
proxy_count_by_state = (
    state_gdf[["state_code"]]
    .merge(
        composting_proxy_gdf["state_code"].value_counts().rename("proxy_count"),
        how="left",
        left_on="state_code",
        right_index=True,
    )
    .sort_values("proxy_count")
)
display(proxy_count_by_state)
print(f"total number of composting proxies: {composting_proxy_gdf.shape[0]:,}")
print(
    "do all states have at least 1 proxy? "
    f"{proxy_count_by_state['proxy_count'].gt(0).all()}"
)
# %% MAP PROXY DATA --------------------------------------------------------------------
ax = composting_proxy_gdf.drop_duplicates().plot(
    "state_code", categorical=True, cmap="Set2", figsize=(10, 10)
)
state_gdf.boundary.plot(ax=ax, color="xkcd:slate", lw=0.2, zorder=1)

# %% STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES ---------------------------
allocated_emis_gdf = allocate_emissions_to_proxy(
    composting_proxy_gdf, EPA_state_emi_df, proxy_has_year=False, use_proportional=False
)
allocated_emis_gdf

# %% STEP 4.1: QC PROXY ALLOCATED EMISSIONS BY STATE AND YEAR --------------------------
proxy_qc_result = QC_proxy_allocation(allocated_emis_gdf, EPA_state_emi_df)

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
    data=EPA_state_emi_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)
proxy_qc_result

# %% STEP 5: RASTERIZE THE CH4 KT AND FLUX ---------------------------------------------
ch4_kt_result_rasters, ch4_flux_result_rasters = grid_allocated_emissions(
    allocated_emis_gdf
)

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
