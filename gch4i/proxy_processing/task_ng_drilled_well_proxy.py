"""
Name:                   task_ng_drilled_well_proxy.py
Date Last Modified:     2025-01-30
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Mapping of natural gas drilled well proxy emissions.
Input Files:            State Geo: global_data_dir_path / "tl_2020_us_state.zip"
                        Processed and Cleaned Enverus Prism & DI: sector_data_dir_path 
                            / "enverus/production/intermediate_outputs/formatted_raw_enverus_tempoutput_{iyear}.csv"
                        NEI: sector_data_dir_path / "nei_og"
                        Emissions: emi_data_dir_path / "gas_well_drilled_emi.csv"
Output Files:           proxy_data_dir_path / "ng_drilled_well_proxy.parquet"
"""

# %% Import Libraries
from pathlib import Path
import os
from typing import Annotated

import pandas as pd
import geopandas as gpd
import numpy as np

from pytask import Product, task, mark

from gch4i.config import (
    proxy_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    emi_data_dir_path,
    years,
)


from gch4i.proxy_processing.ng_oil_production_utils import (
    calc_enverus_rel_emi,
    enverus_df_to_gdf,
    nei_data_years,
    get_nei_file_name,
    ng_spud_count_file_names,
    get_raw_NEI_data,
    create_alt_proxy,
)

# %% Pytask Function


@mark.persist
@task(id="ng_drilled_well_proxy")
def task_get_ng_drilled_well_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    intermediate_outputs_path: Path = sector_data_dir_path / "enverus/production/intermediate_outputs",
    nei_path: Path = sector_data_dir_path / "nei_og",
    ng_drilled_well_emi_path: Path = emi_data_dir_path / "gas_well_drilled_emi.csv",
    drilled_well_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_drilled_well_proxy.parquet",
):
    """
    Data come from Enverus, both Drilling Info and Prism
    The reason 2 datasets are used is because Prism does not include all states
    So remaining states, or those with more DI coverage are taken from DI

    DI: KS, MD, MI, MO, OK, TN

    Prism: AK, AL, AR, AZ, CA, CAO (California Offshore), CO, FL, KY, LA, MS, MT, ND,
    NE, NGOM (federal offshore waters in the Gulf of Mexico), NM, NV, NY, OH, OR, PA,
    SD, TX, UT, VA, WV, WY

    States with no Enverus Data: CT, DE, DC, GA, HI, ID, IL*, IN*, IA, ME, MA, MN, NH,
    NJ, NC, RI, SC, VT, WA, WI, US territories. These states are assumed to have no oil
    and gas production with an exception for IL and IN.

    *IL and IN do not report to Enverus, but do have oil and gas production. Production
    data is taken from the Energy Information Administration (EIA).

    """

    # Load in State ANSI data
    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .reset_index(drop=True)
        .to_crs(4326)
    )

    """
    Make Annual gridded arrays (maps) of well data (a well will be counted every month
        if there is any production that year).
    Includes NA Wells and Production onshore in the CONUS region source emissions are
        related to the presence of a well and its production status (no emission if no
        production).
    Details: ERG does not include a well in the national count if there is no
        (cummulative) gas production from that well.
    Wells are not considered active for a given year if there is no production data that
        year.
    This may cause wells that are coadmpleted but not yet producing to be dropped from
        the national count.
    ERG has developed their own logic to determine if a well is an HF well or not and
        that result is included in the HF variable in this dataset. This method does not
        rely on the Enverus well 'Producing Status'.
    Well Type (e.g., non-associated oil well) is determined based on annual production
        GOR at that well (CUM OIL/ CUM GAS), but the presence of a well will only be
        included in maps in months where monthly gas prod > 0"
    """

    # Proxy Data Dataframes:
    drilled_well_df = pd.DataFrame()  # Gas wells drilled

    ## Enverus DI and Prism Data: 
    # Read in and query formatted and corrrected Enverus data to create dictionaries of 
    # proxy data (Enverus data is from task_enverus_di_prism_data_processing.py)
    for iyear in years:
        enverus_file_name_iyear = f"formatted_raw_enverus_tempoutput_{iyear}.csv"
        enverus_file_path_iyear = os.path.join(intermediate_outputs_path, enverus_file_name_iyear)
        ng_data_temp = (pd.read_csv(enverus_file_path_iyear, dtype={3:'str', 'spud_year': str, 'first_prod_year': str})
                        .query("STATE_CODE.isin(@state_gdf['state_code'])")
                        .query("OFFSHORE == 'N'")
                        .query("CUM_GAS > 0")
                        .assign(gas_to_oil_ratio=lambda df: df['CUM_GAS']/df['CUM_OIL'])
                        .assign(year=str(iyear))
                        .replace(np.inf, 0)
                        .astype({"spud_year": str, "first_prod_year": str})
                        .query("gas_to_oil_ratio > 100 | GOR_QUAL == 'Gas only'")
                        .dropna(subset=["LATITUDE", "LONGITUDE"])
                        )

        # Include wells in map only for months where there is gas production (emissions ~ when production is occuring)
        for imonth in range(1,13):
            imonth_str = f"{imonth:02}"  # convert to 2-digit months
            year_month_str = str(iyear)+'-'+imonth_str
            gas_prod_str = 'GASPROD_'+imonth_str
            # Onshore data for imonth
            ng_data_imonth_temp = (ng_data_temp
                                   .query(f"{gas_prod_str} > 0")
                                   .assign(year_month=str(iyear)+'-'+imonth_str)
                                   .assign(month=imonth)
                                   )
            ng_data_imonth_temp = (ng_data_imonth_temp[[
                'year', 'month', 'year_month','STATE_CODE','AAPG_CODE_ERG','LATITUDE','LONGITUDE',
                'HF','WELL_COUNT',gas_prod_str,
                'comp_year_month','spud_year','first_prod_year']]
                )
            # Drilled Gas Wells
            drilled_well_imonth = (ng_data_imonth_temp[['year', 'month','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT','HF','spud_year','first_prod_year']]
                                  .rename(columns=lambda x: str(x).lower())
                                  .rename(columns={"well_count":"proxy_data"})
                                  # wells with a spud date or first production date in the current year
                                  .query(f"spud_year == '{iyear}' | first_prod_year == '{iyear}'")
                                  # wells with a spud_year == iyear or if no spud date, first_prod_year == iyear
                                  .query(f"spud_year == '{iyear}' | spud_year == 'NaN' | spud_year == 'nan'")
                                  .drop(columns=['hf', 'spud_year', 'first_prod_year'])
                                  .reset_index(drop=True)
                                  )
            drilled_well_df = pd.concat([drilled_well_df,drilled_well_imonth])
    
    # Delete unused temp data
    del ng_data_temp
    del ng_data_imonth_temp
    del drilled_well_imonth

    # Convert to a geodataframe
    drilled_well_df = enverus_df_to_gdf(drilled_well_df)

    # Remove data with empty geometries
    drilled_well_df['empty_geometry'] = drilled_well_df.is_empty
    print("Number of total data entries: ", len(drilled_well_df))
    print("Number of data entries with missing geometry: ", len(drilled_well_df.query("empty_geometry == True")))
    drilled_well_df = drilled_well_df.query("empty_geometry == False").drop(columns="empty_geometry").reset_index(drop=True)

    # Calculate relative emissions
    drilled_well_df = calc_enverus_rel_emi(drilled_well_df)

    # NEI Data:
    nei_df = pd.DataFrame()

    for iyear in years:
        nei_data_year = nei_data_years[nei_data_years['year'] == iyear]['nei_data'].values[0]
        # Well Count
        ifile_name = get_nei_file_name(nei_data_year, ng_spud_count_file_names)
        nei_iyear = get_raw_NEI_data(iyear, nei_data_year, ifile_name)
        nei_df = pd.concat([nei_df, nei_iyear])
    
    # Convert NEI Data to GDF and polygon to centroid point
    nei_df = gpd.GeoDataFrame(nei_df, crs=4326)
    nei_df = nei_df.to_crs(3857)  # projected CRS for centroid calculation
    nei_df.loc[:, 'geometry'] = nei_df.loc[:, 'geometry'].centroid
    nei_df = nei_df.to_crs(4326)
    
    # Add NEI Data to Enverus Data
    drilled_well_df = pd.concat([drilled_well_df, nei_df]).astype({'year': int}).reset_index(drop=True)

    # Delete unused temp data
    del nei_iyear
    del nei_df

    # Correct for missing proxy data
    # 1. Find missing state_code-year pairs
    # 2. Check to see if proxy data exists for state in another year
    #   2a. If the data exists, use proxy data from the closest year
    #   2b. If the data does not exist, assign emissions uniformly across the state

    # Read in emissions data and drop states with 0 emissions
    emi_df = (pd.read_csv(ng_drilled_well_emi_path)
              .query("state_code.isin(@state_gdf['state_code'])")
              .query("ghgi_ch4_kt > 0.0")
              )

    # Retrieve unique state codes for emissions without proxy data
    # This step is necessary, as not all emissions data excludes emission-less states
    emi_states = set(emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    proxy_states = set(drilled_well_df[['state_code', 'year']].itertuples(index=False, name=None))

    # Find missing states
    missing_states = emi_states.difference(proxy_states)

    # Add missing states alternative data to grouped_proxy
    proxy_gdf_final = create_alt_proxy(missing_states, drilled_well_df)

    # Output Proxy Parquet Files
    proxy_gdf_final.to_parquet(drilled_well_output_path)

    return None

# %%
