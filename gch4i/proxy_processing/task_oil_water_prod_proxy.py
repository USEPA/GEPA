"""
Name:                   task_oil_water_prod_proxy.py
Date Last Modified:     2025-07-09
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Mapping of oil water production proxy emissions.
Input Files:            State Geo: global_data_dir_path / "tl_2020_us_state.zip"
                        Processed and Cleaned Enverus Prism & DI: sector_data_dir_path 
                            / "enverus/production/intermediate_outputs/formatted_raw_enverus_tempoutput_{iyear}.csv"
                        NEI: sector_data_dir_path / "nei_og"
                        Emissions: emi_data_dir_path / "prod_water_emi.csv"
Output Files:           basin Other: proxy_data_dir_path / "oil_water_prod_proxy.parquet"
"""

# %%
from pathlib import Path
import os
from typing import Annotated

import pandas as pd
import geopandas as gpd
import numpy as np

from pytask import Product, task, mark

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    emi_data_dir_path,
    max_year,
    min_year,
    years,
)

from gch4i.utils import us_state_to_abbrev
from gch4i.proxy_processing.ng_oil_production_utils import (
    calc_enverus_rel_emi,
    enverus_df_to_gdf,
    nei_data_years,
    get_nei_file_name,
    oil_water_prod_file_names,
    get_raw_NEI_data,
    find_closest_year,
    create_alt_proxy,
)

# %%
@mark.persist
@task(id="oil_water_prod_proxy")
def task_get_oil_water_prod_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    intermediate_outputs_path: Path = sector_data_dir_path / "enverus/production/intermediate_outputs",
    nei_path: Path = sector_data_dir_path / "nei_og",
    prod_water_emi_path: Path = emi_data_dir_path / "prod_water_emi.csv",
    oil_all_well_prod_proxy_path: Path = proxy_data_dir_path / "oil_all_well_prod_proxy.parquet",
    water_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_water_prod_proxy.parquet",
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
        (cummulative) oil production from that well.
    Wells are not considered active for a given year if there is no production data that
        year.
    This may cause wells that are coadmpleted but not yet producing to be dropped from
        the national count.
    ERG has developed their own logic to determine if a well is an HF well or not and
        that result is included in the HF variable in this dataset. This method does not
        rely on the Enverus well 'Producing Status'.
    Well Type (e.g., non-associated oil well) is determined based on annual production
        GOR at that well (CUM OIL/ CUM GAS), but the presence of a well will only be
        included in maps in months where monthly oil prod > 0"
    """

    # Proxy Data Dataframes:
    water_prod_df = pd.DataFrame()

    ## Enverus DI and Prism Data: 
    # Read in and query formatted and corrrected Enverus data to create dictionaries of 
    # proxy data (Enverus data is from task_enverus_di_prism_data_processing.py)
    for iyear in years:
        enverus_file_name_iyear = f"formatted_raw_enverus_tempoutput_{iyear}.csv"
        enverus_file_path_iyear = os.path.join(intermediate_outputs_path, enverus_file_name_iyear)
        oil_data_temp = (pd.read_csv(enverus_file_path_iyear, dtype={3:'str', 'spud_year': str, 'first_prod_year': str})
                         .query("STATE_CODE.isin(@state_gdf['state_code'])")
                         .query("OFFSHORE == 'N'")
                         .query("CUM_OIL > 0")
                         .assign(gas_to_oil_ratio=lambda df: df['CUM_GAS']/df['CUM_OIL'])
                         .assign(year=str(iyear))
                         .replace(np.inf, 0)
                         .astype({"spud_year": str, "first_prod_year": str})
                         .query("gas_to_oil_ratio <= 100")
                         .query("GOR_QUAL == 'Liq only' | GOR_QUAL == 'Liq+Gas'")
                         .dropna(subset=["LATITUDE", "LONGITUDE"])
                         )

        # Include wells in map only for months where there is production (emissions ~ when production is occuring)
        for imonth in range(1,13):
            imonth_str = f"{imonth:02}"  # convert to 2-digit months
            year_month_str = str(iyear)+'-'+imonth_str
            oil_prod_str = 'OILPROD_'+imonth_str
            water_prod_str = 'WATERPROD_'+imonth_str
            # Onshore data for imonth
            oil_data_imonth_temp = (oil_data_temp
                                    .query(f"{oil_prod_str} > 0")
                                    .assign(year_month=str(iyear)+'-'+imonth_str)
                                    .assign(month=imonth)
                                    )
            oil_data_imonth_temp = (oil_data_imonth_temp[[
                'year', 'month', 'year_month','STATE_CODE','AAPG_CODE_ERG','LATITUDE','LONGITUDE',
                'HF','WELL_COUNT',oil_prod_str,water_prod_str,
                'comp_year_month','spud_year','first_prod_year']]
                )
            # Water Production
            # Data Source by state defined in Enverus DrillingInfo Processing - Produced
            # Water_2023-11-14_forGridding.xlsx file.
            if iyear < 2016:  # WV uses NEI data
                water_prod_enverus_states = ['AK','AL','AR','AZ','CA','CO','FL','LA',
                                             'MI','MO','MS','MT','ND','NE','NM','NV',
                                             'NY','OH','SD','TX','UT','VA','WY'
                                             ]
                # States using NEI for reference: ['IL','IN','KS','OK','PA','WV']
            else:  # 2016 and beyond; WV uses Enverus data
                water_prod_enverus_states = ['AK','AL','AR','AZ','CA','CO','FL','LA',
                                             'MI','MO','MS','MT','ND','NE','NM','NV',
                                             'NY','OH','SD','TX','UT','VA','WY','WV'
                                             ]  #WV uses Enverus
                # States using NEI for reference: ['IL','IN','KS','OK','PA']
            # Enverus water production for applicable states (NEI water producted will
            # be added in the NEI section of the code below)
            water_prod_imonth = (oil_data_imonth_temp[['year', 'month','year_month','STATE_CODE','LATITUDE','LONGITUDE',water_prod_str]]
                                .query("STATE_CODE.isin(@water_prod_enverus_states)")
                                .assign(proxy_data=lambda df: df[water_prod_str])
                                .drop(columns=[water_prod_str])
                                .query("proxy_data > 0")
                                .rename(columns=lambda x: str(x).lower())
                                .reset_index(drop=True)
                                )
            water_prod_df = pd.concat([water_prod_df,water_prod_imonth])

    # Delete unused temp data
    del oil_data_temp
    del oil_data_imonth_temp
    del water_prod_imonth

    # Convert to a geodataframe
    water_prod_df = enverus_df_to_gdf(water_prod_df)

    # Remove data with empty geometries
    water_prod_df['empty_geometry'] = water_prod_df.is_empty
    print("Number of total data entries: ", len(water_prod_df))
    print("Number of data entries with missing geometry: ", len(water_prod_df.query("empty_geometry == True")))
    water_prod_df = water_prod_df.query("empty_geometry == False").drop(columns="empty_geometry").reset_index(drop=True)

    # Calculate relative emissions and convert to a geodataframe
    water_prod_df = calc_enverus_rel_emi(water_prod_df)
    water_prod_df = water_prod_df.astype({'year': int})

    # NEI Data:
    nei_df = pd.DataFrame()

    for iyear in years:
        nei_data_year = nei_data_years[nei_data_years['year'] == iyear]['nei_data'].values[0]
        ifile_name = get_nei_file_name(nei_data_year, oil_water_prod_file_names)
        nei_iyear = get_raw_NEI_data(iyear, nei_data_year, ifile_name)
        nei_df = pd.concat([nei_df, nei_iyear])
    
    # Convert NEI Data to GDF and polygon to centroid point
    nei_df = gpd.GeoDataFrame(nei_df, crs=4326)
    nei_df = nei_df.to_crs(3857)  # projected CRS for centroid calculation
    nei_df.loc[:, 'geometry'] = nei_df.loc[:, 'geometry'].centroid
    nei_df = nei_df.to_crs(4326)
    
    # Add NEI Data to Enverus Data
    water_prod_df = pd.concat([water_prod_df, nei_df]).astype({'year': int}).reset_index(drop=True)

    # Separate out wells with zero and non-zero water production with location data
    proxy_gdf_final = water_prod_df.query("rel_emi > 0.0").reset_index(drop=True)
    water_prod_df_zero = water_prod_df.query("rel_emi == 0.0").reset_index(drop=True)

    # Delete unused temp data
    del water_prod_df
    del nei_iyear
    del nei_df

    # Correct for missing proxy data
    # 1. Find missing state_code-year pairs
    # 2. Check to see if proxy data exists for state in another year - if the data
    #    exists, use proxy data from the closest year.
    # 3. Check to see if the proxy data exists for the state but is just 0. Use the
    #    the location information and uniformly assign emissions across the locations.
    # 4. Assign proxy data from a different natural gas proxy to the remaining state-year
    #    combinations with missing data (not needed in v3).
    # 5. If state-year combinations are still missing data, assign emissions uniformly
    #    across the state (not needed in v3).

    # Read in emissions data and drop states with 0 emissions
    emi_df = (pd.read_csv(prod_water_emi_path)
                          .query("state_code.isin(@state_gdf['state_code'])")
                          .query("ghgi_ch4_kt > 0.0")
                          )

    # Retrieve unique state codes for emissions without proxy data
    # This step is necessary, as not all emissions data excludes emission-less states
    emi_states = set(emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    proxy_states = set(proxy_gdf_final[['state_code', 'year']].itertuples(index=False, name=None))
    missing_states = list(emi_states.difference(proxy_states))

    if missing_states:
        # List of state-year combinations with 0 water production reported with locations
        zero_water_states = list(set(water_prod_df_zero[['state_code', 'year']].itertuples(index=False, name=None)))  # state codes with 0 water production in at least one location
        proxy_unique_states = proxy_gdf_final['state_code'].unique()  # state codes covered in proxy
        missing_states_unique = pd.DataFrame(emi_states.difference(proxy_states))[0].unique()  # state codes with at least one year of missing data
        oil_all_well_prod_proxy = gpd.read_parquet(oil_all_well_prod_proxy_path).query("state_code.isin(@missing_states_unique)")
        for imissing_state in range(0, len(missing_states)):
            istate_year = missing_states[imissing_state]
            istate = missing_states[imissing_state][0]
            iyear = missing_states[imissing_state][1]
            # If the state-year combination appears in the list of states that have
            # locations for wells with 0 water production, uniformly assign the emissions
            # across these facilities.
            if istate_year in list(zero_water_states):
                iproxy_data = water_prod_df_zero.query("state_code == @istate").query("year == @iyear").assign(water = 1)
                iproxy_data['rel_emi'] = iproxy_data.groupby(["state_code", "year_month"])['water'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
                iproxy_data['annual_rel_emi'] = iproxy_data.groupby(["state_code", "year"])['water'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
                iproxy_data = iproxy_data.drop(columns='water')
                proxy_gdf_final = pd.concat([proxy_gdf_final, iproxy_data]).reset_index(drop=True)
                print(f"({istate}, {iyear}) has been updated with locations with 0 water production in state and year.")
            # If the missing state code-year pair has data for another year, assign
            # the proxy data for the next available previous year
            elif istate in proxy_unique_states:
                # Get proxy data for the state for all years
                iproxy_data = (proxy_gdf_final
                               .query("state_code == @istate")
                               .reset_index(drop=True)
                               )
                # Get years that have proxy data
                iproxy_unique_years = iproxy_data['year'].unique()
                # Find the closest year to the missing proxy year
                iyear_closest = find_closest_year(iproxy_unique_years, iyear)
                # Assign proxy data of the closest year to the missing proxy year
                iproxy_data = (iproxy_data
                          .query("year == @iyear_closest")
                          .assign(year=iyear)
                          .reset_index(drop=True)
                          )
                # Update year_month column to be the correct year
                for ifacility in np.arange(0, len(iproxy_data)):
                    imonth_str = str(iproxy_data['year_month'][ifacility][5:8])
                    iyear_month_str = str(iyear)+'-'+imonth_str
                    iproxy_data.loc[ifacility, 'year_month'] = iyear_month_str
                    iproxy_data.loc[ifacility, 'month'] = int(imonth_str)
                proxy_gdf_final = gpd.GeoDataFrame(pd.concat([proxy_gdf_final, iproxy_data], ignore_index=True))
                print(f"({istate}, {iyear}) has been updated with {iyear_closest} data.")
            # Use proxy data from oil_all_well_prod_proxy
            else:
                iproxy_data = oil_all_well_prod_proxy.query("state_code == @istate").query("year == @iyear")
                proxy_gdf_final = gpd.GeoDataFrame(pd.concat([proxy_gdf_final, iproxy_data], ignore_index=True))
                print(f"({istate}, {iyear}) emissions have been assigned using oil_all_well_prod_proxy.")
          
    # Check for missing states after applying the closest year data to states with proxy data in 2012-2022
    proxy_states = set(proxy_gdf_final[['state_code', 'year']].itertuples(index=False, name=None))
    missing_states = emi_states.difference(proxy_states)

    # Check that annual relative emissions sum to 1.0 each state/year combination
    sums_annual = proxy_gdf_final.groupby(["state_code", "year"])["annual_rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums_annual, 1.0, atol=1e-8).all(), f"Annual relative emissions do not sum to 1 for each year and state; {sums_annual}"  # assert that the sums are close to 1

    # Check that monthly relative emissions sum to 1.0 each state/year_month combination
    sums_monthly = proxy_gdf_final.groupby(["state_code", "year_month"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums_monthly, 1.0, atol=1e-8).all(), f"Monthly relative emissions do not sum to 1 for each year_month and state; {sums_monthly}"  # assert that the sums are close to 1

    # Output Proxy Parquet Files
    proxy_gdf_final.to_parquet(water_prod_output_path)

    return None

# %%
