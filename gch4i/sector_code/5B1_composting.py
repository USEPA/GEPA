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
NOTE: THIS IS NOT A COMPLETE SCRIPT. IT IS A SHELL BEING KEPT UNTIL WE KNOW COMPOSTING
FITS INTO THE NEW STANDARD GRIDDER FRAMEWORK. THIS SCRIPT IS NOT CURRENTLY BEING USED.
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


from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
    emi_data_dir_path,
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


# TODO: move to emis file
@mark.persist
@task(id="composting_emi")
def task_composting_emi(input_path, output_path):
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
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
        # .query("state_code.isin(@state_gdf['state_code'])")
    )
    emi_df.to_csv(output_path)
    return emi_df





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


# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# INVENTORY INPUT FILE
inventory_workbook_path = composting_dir / "State_Composting_1990-2021.xlsx"

# PROXY INPUT FILES
# state vector refence with state_code
state_geo_path = 
# input 1: excess food opportunities data
excess_food_op_path = 
# input 2: frs facilities where NAICS code is for composting
frs_naics_path = 
frs_facility_path = 
# input 3: biocycle facility locations pulled from v2
biocycle_path = 
# input 4: ad council data from their kml file
comp_council_path = 

# the NAICS code pulled from the v2 notebook for facilities in the FRS data

# the spatial tolerance for removing duplicate facility points

# %% STEP 1. Load GHGI-Proxy Mapping Files ---------------------------------------------

# %% STEP 2: Read In EPA State GHGI Emissions by Year ----------------------------------

# Get state vectors and state_code for use with inventory and proxy data


EPA_state_emi_df = get_composting_inventory_data(inventory_workbook_path)

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
    y="ghgi_ch4_kt",
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
