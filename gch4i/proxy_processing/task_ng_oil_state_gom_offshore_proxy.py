# %%
from pathlib import Path
import os
from typing import Annotated

from pyarrow import parquet
import pandas as pd
import geopandas as gpd
import numpy as np
from pytask import Product, task, mark

from gch4i.config import (
    global_data_dir_path,
    sector_data_dir_path,
    proxy_data_dir_path,
    emi_data_dir_path,
    min_year,
    max_year,
    years,
)
from gch4i.proxy_processing.ng_oil_production_utils import (
    calc_enverus_rel_emi,
    enverus_df_to_gdf,
    create_alt_proxy,
)

# %%
@mark.persist
@task(id="ng_oil_state_gom_offshore_proxy")
def task_get_ng_oil_state_gom_offshore_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_well_counts_path: Path = sector_data_dir_path / "enverus/production/temp_data_v2/Enverus DrillingInfo Processing - Well Counts_2021-03-17.xlsx",
    enverus_production_path: Path = sector_data_dir_path / "enverus/production",
    intermediate_outputs_path: Path = sector_data_dir_path / "enverus/production/intermediate_outputs",
    oil_gom_state_emi_path: Path = emi_data_dir_path / "oil_gom_state_emi.csv",
    trans_offshore_emi_path: Path = emi_data_dir_path / "trans_offshore_emi.csv",
    oil_pac_federal_state_emi_path: Path = emi_data_dir_path / "oil_pac_federal_state_emi.csv",
    state_gom_offshore_emi_path: Path = emi_data_dir_path / "state_gom_offshore_emi.csv",
    oil_state_gom_offshore_output_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_state_gom_offshore_well_count_proxy.parquet",
    oil_pac_fed_state_output_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_pac_fed_state_proxy.parquet",
    ng_state_gom_offshore_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_state_gom_offshore_well_count_proxy.parquet",
    ):
    """
    Data come from Enverus, Prism only. Drilling Info data is not used for the offshore
    well data because DI was only used for KS, MD, MI, MO, OK, and TN which are not
    in the offshore region of the U.S.

    States to produce offshore data: AL, CA, CAO (California Offshore), FL, LA, MS, TX
    Note that there is no Enverus Prism data for FL and MS.
    
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

    # Well and Production Data (from Enverus)
    # Read In and Format DI data
    # 1. Read Data
    # 2. Drop unused columns, rename columns
    # 3. Calculate annual cummulate production totals
    # 4. Save the data as a year-specific variable

    # Based on ERGs logic, active wells are determined based on their production levels and not producing status
    Enverus_data_dict = {}

    for iyear in years:
        # Prism Data
        Prism_file_name = f"prism_monthly_wells_offshore_{iyear}.xlsx"
        Prism_file_path = os.path.join(enverus_production_path, Prism_file_name)
        Prism_data = (pd.read_excel(
            Prism_file_path,
            usecols={'STATE', 'LATITUDE', 'LONGITUDE', 'OFFSHORE', 'GOR_QUAL',
                     'LIQUIDSPROD_BBL_01', 'GASPROD_MCF_01', 'WATERPROD_BBL_01',
                     'LIQUIDSPROD_BBL_02', 'GASPROD_MCF_02', 'WATERPROD_BBL_02',
                     'LIQUIDSPROD_BBL_03', 'GASPROD_MCF_03', 'WATERPROD_BBL_03',
                     'LIQUIDSPROD_BBL_04', 'GASPROD_MCF_04', 'WATERPROD_BBL_04',
                     'LIQUIDSPROD_BBL_05', 'GASPROD_MCF_05', 'WATERPROD_BBL_05',
                     'LIQUIDSPROD_BBL_06', 'GASPROD_MCF_06', 'WATERPROD_BBL_06',
                     'LIQUIDSPROD_BBL_07', 'GASPROD_MCF_07', 'WATERPROD_BBL_07',
                     'LIQUIDSPROD_BBL_08', 'GASPROD_MCF_08', 'WATERPROD_BBL_08',
                     'LIQUIDSPROD_BBL_09', 'GASPROD_MCF_09', 'WATERPROD_BBL_09',
                     'LIQUIDSPROD_BBL_10', 'GASPROD_MCF_10', 'WATERPROD_BBL_10',
                     'LIQUIDSPROD_BBL_11', 'GASPROD_MCF_11', 'WATERPROD_BBL_11',
                     'LIQUIDSPROD_BBL_12', 'GASPROD_MCF_12', 'WATERPROD_BBL_12',
                     })
                     .rename(columns={'STATE':'STATE_CODE',
                            'LIQUIDSPROD_BBL_01':'OILPROD_01','GASPROD_MCF_01':'GASPROD_01','WATERPROD_BBL_01':'WATERPROD_01',
                            'LIQUIDSPROD_BBL_02':'OILPROD_02','GASPROD_MCF_02':'GASPROD_02','WATERPROD_BBL_02':'WATERPROD_02',
                            'LIQUIDSPROD_BBL_03':'OILPROD_03','GASPROD_MCF_03':'GASPROD_03','WATERPROD_BBL_03':'WATERPROD_03',
                            'LIQUIDSPROD_BBL_04':'OILPROD_04','GASPROD_MCF_04':'GASPROD_04','WATERPROD_BBL_04':'WATERPROD_04',
                            'LIQUIDSPROD_BBL_05':'OILPROD_05','GASPROD_MCF_05':'GASPROD_05','WATERPROD_BBL_05':'WATERPROD_05',
                            'LIQUIDSPROD_BBL_06':'OILPROD_06','GASPROD_MCF_06':'GASPROD_06','WATERPROD_BBL_06':'WATERPROD_06',
                            'LIQUIDSPROD_BBL_07':'OILPROD_07','GASPROD_MCF_07':'GASPROD_07','WATERPROD_BBL_07':'WATERPROD_07',
                            'LIQUIDSPROD_BBL_08':'OILPROD_08','GASPROD_MCF_08':'GASPROD_08','WATERPROD_BBL_08':'WATERPROD_08',
                            'LIQUIDSPROD_BBL_09':'OILPROD_09','GASPROD_MCF_09':'GASPROD_09','WATERPROD_BBL_09':'WATERPROD_09',
                            'LIQUIDSPROD_BBL_10':'OILPROD_10','GASPROD_MCF_10':'GASPROD_10','WATERPROD_BBL_10':'WATERPROD_10',
                            'LIQUIDSPROD_BBL_11':'OILPROD_11','GASPROD_MCF_11':'GASPROD_11','WATERPROD_BBL_11':'WATERPROD_11',
                            'LIQUIDSPROD_BBL_12':'OILPROD_12','GASPROD_MCF_12':'GASPROD_12','WATERPROD_BBL_12':'WATERPROD_12',
                            })
                     .assign(WELL_COUNT=1)
                     .query("OFFSHORE == 'Y'")
                     )
        
        # Replace nans with zeros, and sum annual production
        Prism_data.loc[:, Prism_data.columns.str.contains('GASPROD_')] = Prism_data.loc[:, Prism_data.columns.str.contains('GASPROD_')].fillna(0)
        Prism_data.loc[:, Prism_data.columns.str.contains('OILPROD_')] = Prism_data.loc[:, Prism_data.columns.str.contains('OILPROD_')].fillna(0)
        Prism_data.loc[:, Prism_data.columns.str.contains('WATERPROD_')] = Prism_data.loc[:, Prism_data.columns.str.contains('WATERPROD_')].fillna(0)

        # Calculate cummulative annual production totals for Gas, Oil, Water
        Prism_data['CUM_GAS'] = Prism_data.loc[:,Prism_data.columns.str.contains('GASPROD_')].sum(1)
        Prism_data['CUM_OIL'] = Prism_data.loc[:,Prism_data.columns.str.contains('OILPROD_')].sum(1)
        Prism_data['CUM_WATER'] = Prism_data.loc[:,Prism_data.columns.str.contains('WATERPROD_')].sum(1)
                
        # Save out the data for that year
        Enverus_data_dict[f'{iyear}'] = Prism_data

        del Prism_data

    # Correct Enverus Data for Select States

    # 1) Read In Coverage Table from State Well Counts File from ERG
    # (specifies the first year with bad data and which years need to be corrected; 
    # all years including and after the first bad year of data need to be corrected)

    ERG_StateWellCounts_LastGoodDataYear = (pd.read_excel(
        enverus_well_counts_path,
        sheet_name = "2021 - Coverage",
        usecols = {"State","Last Good Year"},
        skiprows = 2,
        nrows = 40)
        )

    # 2) Loops through the each state and year in Enverus to determine if the data for that particualar year needs to 
    # be corrected. At the moment, the only corrections ERG makes to the data is to use the prior year of data if there
    # is no new Enverus data reportd for that state. If a particular state is not included for any years in the Enverus
    # dataset, then a row of zeros is added to the Enverus table for that year.

    offshore_states = ['AL', 'CAO', 'FL', 'LA', 'MS', 'TX']
        
    for istate in np.arange(0,len(offshore_states)):
        correctdata = 0
        istate_code = offshore_states[istate]
        lastgoodyear = ERG_StateWellCounts_LastGoodDataYear['Last Good Year'][ERG_StateWellCounts_LastGoodDataYear['State'] == istate_code].values
        if lastgoodyear  == max_year:
            lastgoodyear = max_year+5 #if state isn't included in correction list, don't correct any data
        
        for iyear in years:
            enverus_data_temp= Enverus_data_dict[f'{iyear}'].copy()
            state_list = np.unique(enverus_data_temp['STATE_CODE'])
            if istate_code in state_list:
                inlist =1
            else:
                inlist = 0
            if inlist ==1 or correctdata==1: #if the state is included in Enverus data, or had data for at least one good year
                #if first year, correctdata will be zero, but inlist will also be zero if no Enverus data
                #check to see whether corrections are necessary for the given year/state
                if iyear == (lastgoodyear):
                    print(istate_code,iyear,'last good year')
                    # This is the last year of good data. Do not correct the data but save
                    # but so that this data can be used for all following years for that state
                    temp_data = enverus_data_temp[enverus_data_temp['STATE_CODE'] == istate_code]
                    correctdata=1
                elif iyear > lastgoodyear: 
                    print(istate_code,iyear)
                    # correct data for all years equal to and after the first bad year (remove old data first if necessary)
                    if inlist == 1:
                        enverus_data_temp = enverus_data_temp[enverus_data_temp['STATE_CODE'] != istate_code]
                    enverus_data_temp = pd.concat([enverus_data_temp,temp_data],ignore_index=True)
                    print(istate_code +' data for ' +str(iyear) +' were corrected with '+str(lastgoodyear)+' data')
                else:
                    no_corrections =1
                    
            if inlist == 0 and correctdata == 0:
            # if there is no Enverus data for a given state, and there was no good data, add a row with default values
                print(istate_code + ' has no Enverus data in the year ' + str(iyear))
                
            # save that year of Enverus data
            enverus_data_temp.reset_index(drop=True, inplace=True)
            Enverus_data_dict[f'{iyear}'] = enverus_data_temp.copy()
            tempoutput_filename = f'formatted_raw_enverus_offshore_tempoutput_{iyear}.csv'
            tempoutput_filepath = os.path.join(intermediate_outputs_path, tempoutput_filename)
            enverus_data_temp.to_csv(tempoutput_filepath, index=False)
        
    # create proxy files
    ng_state_gom_offshore_df = pd.DataFrame()
    oil_state_gom_offshore_df = pd.DataFrame()
    oil_pac_fed_state_df = pd.DataFrame()

    # ng proxy
    for iyear in years:
        ng_data_temp = (Enverus_data_dict[f'{iyear}']
                        .query("STATE_CODE.isin(@offshore_states)")
                        .query("CUM_GAS > 0")
                        .assign(gas_to_oil_ratio=lambda df: df['CUM_GAS']/df['CUM_OIL'])
                        .assign(year=str(iyear))
                        .replace(np.inf, 0)
                        .query("gas_to_oil_ratio > 100 | GOR_QUAL == 'Gas only'")
                        .dropna(subset=["LATITUDE", "LONGITUDE"])
                        )
        # Include wells in map only for months where there is gas production (emissions ~ when production is occuring)
        for imonth in range(1, 13):
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
                'year', 'month', 'year_month', 'STATE_CODE', 'LATITUDE', 'LONGITUDE',
                'WELL_COUNT', gas_prod_str,]]
                )
            # State GOM Offshore Gas Well Count
            ng_state_gom_offshore_imonth = (ng_data_imonth_temp[['year', 'month', 'year_month', 'STATE_CODE', 'LATITUDE', 'LONGITUDE', 'WELL_COUNT']]
                                            .rename(columns=lambda x: str(x).lower())
                                            .rename(columns={"well_count": "proxy_data"})
                                            .reset_index(drop=True)
                                            )
            ng_state_gom_offshore_df = pd.concat([ng_state_gom_offshore_df, ng_state_gom_offshore_imonth])

    # oil proxies
    for iyear in years:
        oil_data_temp = (Enverus_data_dict[f'{iyear}']
                         .query("STATE_CODE.isin(@offshore_states)")
                         .query("CUM_OIL > 0")
                         .assign(gas_to_oil_ratio=lambda df: df['CUM_GAS']/df['CUM_OIL'])
                         .assign(year=str(iyear))
                         .replace(np.inf, 0)
                         .query("gas_to_oil_ratio <= 100")
                         .query("GOR_QUAL == 'Liq only' | GOR_QUAL == 'Liq+Gas'")
                         .dropna(subset=["LATITUDE", "LONGITUDE"])
                         )
        # Include wells in map only for months where there is oil production (emissions ~ when production is occuring)
        for imonth in range(1,13):
            imonth_str = f"{imonth:02}"  # convert to 2-digit months
            year_month_str = str(iyear)+'-'+imonth_str
            oil_prod_str = 'OILPROD_'+imonth_str
            # Onshore data for imonth
            oil_data_imonth_temp = (oil_data_temp
                                    .query(f"{oil_prod_str} > 0")
                                    .assign(year_month=str(iyear)+'-'+imonth_str)
                                    .assign(month=imonth)
                                    )
            oil_data_imonth_temp = (oil_data_imonth_temp[[
                'year', 'month', 'year_month', 'STATE_CODE', 'LATITUDE', 'LONGITUDE',
                'WELL_COUNT', oil_prod_str,]]
                )
            # State GOM Offshore Oil Well Count
            oil_state_gom_offshore_imonth = (oil_data_imonth_temp[['year', 'month', 'year_month', 'STATE_CODE', 'LATITUDE', 'LONGITUDE', 'WELL_COUNT']]
                                             .rename(columns=lambda x: str(x).lower())
                                             .rename(columns={"well_count": "proxy_data"})
                                             .query("state_code != 'CAO'")
                                             .reset_index(drop=True)
                                             )
            oil_state_gom_offshore_df = pd.concat([oil_state_gom_offshore_df, oil_state_gom_offshore_imonth])
            # Pacific Federal State Offshore Oil Well Count
            oil_pac_fed_state_imonth = (oil_data_imonth_temp[['year', 'month', 'year_month', 'STATE_CODE', 'LATITUDE', 'LONGITUDE', 'WELL_COUNT']]
                                        .rename(columns=lambda x: str(x).lower())
                                        .rename(columns={"well_count": "proxy_data"})
                                        .query("state_code == 'CAO'")
                                        .reset_index(drop=True)
                                        )
            oil_pac_fed_state_df = pd.concat([oil_pac_fed_state_df, oil_pac_fed_state_imonth])

    # Calculate relative emissions and convert to a geodataframe
    ng_state_gom_offshore_df = calc_enverus_rel_emi(ng_state_gom_offshore_df)
    ng_state_gom_offshore_df = enverus_df_to_gdf(ng_state_gom_offshore_df)
    ng_state_gom_offshore_df = ng_state_gom_offshore_df.astype({'year': int})
    oil_state_gom_offshore_df = calc_enverus_rel_emi(oil_state_gom_offshore_df)
    oil_state_gom_offshore_df = enverus_df_to_gdf(oil_state_gom_offshore_df)
    oil_state_gom_offshore_df = oil_state_gom_offshore_df.astype({'year': int})
    # Note that pacific federal state is treated as one region, so normalization is done without the state_code
    oil_pac_fed_state_df['annual_rel_emi'] = (
        oil_pac_fed_state_df.groupby(["year"])['proxy_data']
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    )
    oil_pac_fed_state_df['rel_emi'] = (
        oil_pac_fed_state_df.groupby(["year_month"])['proxy_data']
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    )
    oil_pac_fed_state_df = oil_pac_fed_state_df.drop(columns='proxy_data')
    oil_pac_fed_state_df = enverus_df_to_gdf(oil_pac_fed_state_df)
    oil_pac_fed_state_df = oil_pac_fed_state_df.astype({'year': int})

    # Correct for missing proxy data
    # 1. Find missing state_code-year pairs
    # 2. Check to see if proxy data exists for state in another year
    #   2a. If the data exists, use proxy data from the closest year
    #   2b. If the data does not exist, assign emissions uniformly across the state

    # Read in emissions data and drop states with 0 emissions
    ng_emi_df = (pd.read_csv(state_gom_offshore_emi_path)
                          .query("state_code.isin(@state_gdf['state_code'])")
                          .query("ghgi_ch4_kt > 0.0")
                          )
    oil_gom_df = (pd.read_csv(oil_gom_state_emi_path)
                       .query("state_code.isin(@state_gdf['state_code'])")
                       .query("ghgi_ch4_kt > 0.0")
                       )
    oil_trans_df = (pd.read_csv(trans_offshore_emi_path)
                        .query("state_code.isin(@state_gdf['state_code'])")
                        .query("ghgi_ch4_kt > 0.0")
                        )
    oil_gom_emi_df = pd.concat([oil_gom_df, oil_trans_df]).reset_index(drop=True)

    # Retrieve unique state codes for emissions without proxy data
    # This step is necessary, as not all emissions data excludes emission-less states
    ng_emi_states = set(ng_emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    ng_proxy_states = set(ng_state_gom_offshore_df[['state_code', 'year']].itertuples(index=False, name=None))
    oil_gom_emi_states = set(oil_gom_emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    oil_gom_proxy_states = set(oil_state_gom_offshore_df[['state_code', 'year']].itertuples(index=False, name=None))

    # Find missing states
    ng_missing_states = ng_emi_states.difference(ng_proxy_states)
    oil_gom_missing_states = oil_gom_emi_states.difference(oil_gom_proxy_states)

    # Add missing states alternative data to grouped_proxy
    ng_proxy_gdf_final = create_alt_proxy(ng_missing_states, ng_state_gom_offshore_df)
    oil_gom_proxy_gdf_final = create_alt_proxy(oil_gom_missing_states, oil_state_gom_offshore_df)

    # Output Proxy Parquet Files
    ng_proxy_gdf_final.to_parquet(ng_state_gom_offshore_output_path)
    oil_gom_proxy_gdf_final.to_parquet(oil_state_gom_offshore_output_path)
    oil_pac_fed_state_df.to_parquet(oil_pac_fed_state_output_path)

    return None

# %%
