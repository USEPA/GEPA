"""
Name:                   task_enverus_di_prism_data_processing.py
Date Last Modified:     2025-01-30
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Processes and cleans the Enverus DI and Prism data stored on
                            disk and read into natural gas and petroleum proxies. This
                            script should be run before any other proxies are created.
Input Files:            Enverus DI: sector_data_dir_path / "enverus/production/didsk_monthly_{iyear}.csv"
                        Enverus Prism: sector_data_dir_path / "enverus/production/prism_monthly_{iyear}.csv"
                        Enverus Data Coverage: sector_data_dir_path / "enverus/production/temp_data_v2/Enverus DrillingInfo Processing - Well Counts_2021-03-17.xlsx"
Output Files:           Formatted and Corrected Enverus DI and Prism Data: sector_data_dir_path / "enverus/production/intermediate_outputs/formatted_raw_enverus_tempoutput_{iyear}.csv"
"""

# %% Import Libraries
from pathlib import Path
import os


import pandas as pd
import geopandas as gpd
import numpy as np
from pytask import task, mark

from gch4i.config import (
    global_data_dir_path,
    sector_data_dir_path,
    max_year,
    years,
)

# %% Pytask Functions


@mark.persist
@task(id="enverus_di_prism_data_processing")
def task_get_enverus_di_prism_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_production_path: Path = sector_data_dir_path / "enverus/production",
    intermediate_outputs_path: Path = sector_data_dir_path / "enverus/production/intermediate_outputs",
    enverus_well_counts_path: Path = sector_data_dir_path / "enverus/production/temp_data_v2/Enverus DrillingInfo Processing - Well Counts_2021-03-17.xlsx"
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

    # Well and Production Data (from Enverus)
    # Read In & Combine Each Year of Prism & DI Monthly Data (from Enverus)

    # Data come from Enverus, both Drilling Info and Prism
    # The reason 2 datasets are used is because Prism does not include all states
    # So remaining states, or those with more DI coverage are taken from DI

    # Read In and Format the Prism and DI data
    # 1. Read Data
    # 2. Drop unused columns, rename columns to match between DI and Prism
    # 3. Combine DI and Prism into one data array
    # 4. Calculate annual cummulate production totals
    # 5. Save the data as a year-specific variable

    # Based on ERGs logic, active wells are determined based on their production levels and not producing status
    Enverus_data_dict = {}
    DI_data_dict = {}
    Prism_data_dict = {}
    for iyear in years:
        # DI data
        DI_file_name = f"didsk_monthly_{iyear}.csv"
        DI_file_path = os.path.join(enverus_production_path, DI_file_name)
        DI_data = (pd.read_csv(
            DI_file_path,
            usecols=['WELL_COUNT_ID', 'STATE', 'COUNTY', 'BASIN', 'AAPG_CODE_ERG',
                     'LATITUDE', 'LONGITUDE', 'STATUS', 'COMPDATE',
                     'SPUDDATE', 'FIRSTPRODDATE', 'HF', 'OFFSHORE', 'GOR',
                     'GOR_QUAL', 'PROD_FLAG', 'PRODYEAR',
                     'LIQ_01', 'GAS_01', 'WTR_01', 'LIQ_02', 'GAS_02', 'WTR_02',
                     'LIQ_03', 'GAS_03', 'WTR_03', 'LIQ_04', 'GAS_04', 'WTR_04',
                     'LIQ_05', 'GAS_05', 'WTR_05', 'LIQ_06', 'GAS_06', 'WTR_06',
                     'LIQ_07', 'GAS_07', 'WTR_07', 'LIQ_08', 'GAS_08', 'WTR_08',
                     'LIQ_09', 'GAS_09', 'WTR_09', 'LIQ_10', 'GAS_10', 'WTR_10',
                     'LIQ_11', 'GAS_11', 'WTR_11', 'LIQ_12', 'GAS_12', 'WTR_12',],
            dtype={7: 'str'})
            .rename(columns={
                'WELL_COUNT_ID': 'WELL_COUNT', 'STATE': 'STATE_CODE',
                'STATUS': 'PRODUCING_STATUS',
                'LIQ_01': 'OILPROD_01', 'GAS_01': 'GASPROD_01', 'WTR_01': 'WATERPROD_01',
                'LIQ_02': 'OILPROD_02', 'GAS_02': 'GASPROD_02', 'WTR_02': 'WATERPROD_02',
                'LIQ_03': 'OILPROD_03', 'GAS_03': 'GASPROD_03', 'WTR_03': 'WATERPROD_03',
                'LIQ_04': 'OILPROD_04', 'GAS_04': 'GASPROD_04', 'WTR_04': 'WATERPROD_04',
                'LIQ_05': 'OILPROD_05', 'GAS_05': 'GASPROD_05', 'WTR_05': 'WATERPROD_05',
                'LIQ_06': 'OILPROD_06', 'GAS_06': 'GASPROD_06', 'WTR_06': 'WATERPROD_06',
                'LIQ_07': 'OILPROD_07', 'GAS_07': 'GASPROD_07', 'WTR_07': 'WATERPROD_07',
                'LIQ_08': 'OILPROD_08', 'GAS_08': 'GASPROD_08', 'WTR_08': 'WATERPROD_08',
                'LIQ_09': 'OILPROD_09', 'GAS_09': 'GASPROD_09', 'WTR_09': 'WATERPROD_09',
                'LIQ_10': 'OILPROD_10', 'GAS_10': 'GASPROD_10', 'WTR_10': 'WATERPROD_10',
                'LIQ_11': 'OILPROD_11', 'GAS_11': 'GASPROD_11', 'WTR_11': 'WATERPROD_11',
                'LIQ_12': 'OILPROD_12', 'GAS_12': 'GASPROD_12', 'WTR_12': 'WATERPROD_12'
                })
            # TODO: Check to see if this should actually be set to 1
            .assign(WELL_COUNT=1)
            )
        # Format completion date (YYYY-MM)
        for iwell in range(0, len(DI_data)):
            comp_date = str(DI_data.loc[iwell, 'COMPDATE'])
            if comp_date == 'NaN':
                comp_year_month = 'NaN'
            elif comp_date == 'nan':
                comp_year_month = 'NaN'
            else:  # date format M/DD/YYYY
                comp_month = f"{int(comp_date.split('/')[0]):02}"
                comp_year = f"{int(comp_date.split('/')[2])}"
                comp_year_month = str(comp_year)+'-'+str(comp_month)
            DI_data.loc[iwell, 'comp_year_month'] = comp_year_month
        # Format spud date (YYYY)
        for iwell in range(0, len(DI_data)):
            spud_date = str(DI_data.loc[iwell, 'SPUDDATE'])
            if spud_date == 'NaN':
                spud_year = 'NaN'
            elif spud_date == 'nan':
                spud_year = 'NaN'
            else:  # date format M/DD/YYYY
                spud_year = f"{int(spud_date.split('/')[2])}"
                spud_year = str(spud_year)
            DI_data.loc[iwell, 'spud_year'] = spud_year
        # Format first production date (YYYY)
        for iwell in range(0, len(DI_data)):
            first_prod_date = str(DI_data.loc[iwell, 'FIRSTPRODDATE'])
            if first_prod_date == 'NaN':
                first_prod_year = 'NaN'
            elif first_prod_date == 'nan':
                first_prod_year = 'NaN'
            else:  # date format M/DD/YYYY
                first_prod_year = f"{int(first_prod_date.split('/')[2])}"
                first_prod_year = str(first_prod_year)
            DI_data.loc[iwell, 'first_prod_year'] = first_prod_year
        DI_data_dict[f'{iyear}'] = DI_data

        # Prism Data
        Prism_file_name = f"prism_monthly_{iyear}.csv"
        Prism_file_path = os.path.join(enverus_production_path, Prism_file_name)
        Prism_data = (pd.read_csv(
            Prism_file_path,
            usecols=[
                'STATE', 'COUNTY', 'ENVBASIN', 'AAPG_CODE_ERG',
                'LATITUDE', 'LONGITUDE', 'ENVWELLSTATUS', 'COMPLETIONDATE',
                'SPUDDATE', 'FIRSTPRODDATE', 'HF', 'OFFSHORE', 'GOR',
                'GOR_QUAL', 'PROD_FLAG', 'PRODYEAR',
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
                'LIQUIDSPROD_BBL_12', 'GASPROD_MCF_12', 'WATERPROD_BBL_12'
                ],
            dtype={7: 'str'})
            .rename(columns={
                'STATE': 'STATE_CODE', 'ENVBASIN': 'BASIN',
                'ENVWELLSTATUS': 'PRODUCING_STATUS',
                'COMPLETIONDATE': 'COMPDATE',
                'LIQUIDSPROD_BBL_01': 'OILPROD_01', 'GASPROD_MCF_01': 'GASPROD_01', 'WATERPROD_BBL_01': 'WATERPROD_01',
                'LIQUIDSPROD_BBL_02': 'OILPROD_02', 'GASPROD_MCF_02': 'GASPROD_02', 'WATERPROD_BBL_02': 'WATERPROD_02',
                'LIQUIDSPROD_BBL_03': 'OILPROD_03', 'GASPROD_MCF_03': 'GASPROD_03', 'WATERPROD_BBL_03': 'WATERPROD_03',
                'LIQUIDSPROD_BBL_04': 'OILPROD_04', 'GASPROD_MCF_04': 'GASPROD_04', 'WATERPROD_BBL_04': 'WATERPROD_04',
                'LIQUIDSPROD_BBL_05': 'OILPROD_05', 'GASPROD_MCF_05': 'GASPROD_05', 'WATERPROD_BBL_05': 'WATERPROD_05',
                'LIQUIDSPROD_BBL_06': 'OILPROD_06', 'GASPROD_MCF_06': 'GASPROD_06', 'WATERPROD_BBL_06': 'WATERPROD_06',
                'LIQUIDSPROD_BBL_07': 'OILPROD_07', 'GASPROD_MCF_07': 'GASPROD_07', 'WATERPROD_BBL_07': 'WATERPROD_07',
                'LIQUIDSPROD_BBL_08': 'OILPROD_08', 'GASPROD_MCF_08': 'GASPROD_08', 'WATERPROD_BBL_08': 'WATERPROD_08',
                'LIQUIDSPROD_BBL_09': 'OILPROD_09', 'GASPROD_MCF_09': 'GASPROD_09', 'WATERPROD_BBL_09': 'WATERPROD_09',
                'LIQUIDSPROD_BBL_10': 'OILPROD_10', 'GASPROD_MCF_10': 'GASPROD_10', 'WATERPROD_BBL_10': 'WATERPROD_10',
                'LIQUIDSPROD_BBL_11': 'OILPROD_11', 'GASPROD_MCF_11': 'GASPROD_11', 'WATERPROD_BBL_11': 'WATERPROD_11',
                'LIQUIDSPROD_BBL_12': 'OILPROD_12', 'GASPROD_MCF_12': 'GASPROD_12', 'WATERPROD_BBL_12': 'WATERPROD_12'})
            .assign(WELL_COUNT=1)
            )
        # Format completion date (YYYY-MM)
        for iwell in range(0, len(Prism_data)):
            comp_date = str(Prism_data.loc[iwell, 'COMPDATE'])
            if comp_date == 'NaN':
                comp_year_month = 'NaN'
            elif comp_date == 'nan':
                comp_year_month = 'NaN'
            else:  # date format YYYY-MM-DD
                comp_month = f"{int(comp_date.split('-')[1]):02}"
                comp_year = f"{int(comp_date.split('-')[0])}"
                comp_year_month = str(comp_year)+'-'+str(comp_month)
            Prism_data.loc[iwell, 'comp_year_month'] = comp_year_month
        # Format spud date (YYYY)
        for iwell in range(0, len(Prism_data)):
            spud_date = str(Prism_data.loc[iwell, 'SPUDDATE'])
            if spud_date == 'NaN':
                spud_year = 'NaN'
            elif spud_date == 'nan':
                spud_year = 'NaN'
            else:  # date format YYYY-MM-DD
                spud_year = f"{int(spud_date.split('-')[0])}"
                spud_year = str(spud_year)
            Prism_data.loc[iwell, 'spud_year'] = spud_year
        # Format first production date (YYYY)
        for iwell in range(0, len(Prism_data)):
            first_prod_date = str(Prism_data.loc[iwell, 'FIRSTPRODDATE'])
            if first_prod_date == 'NaN':
                first_prod_year = 'NaN'
            elif first_prod_date == 'nan':
                first_prod_year = 'NaN'
            else:  # date format YYYY-MM-DD
                first_prod_year = f"{int(first_prod_date.split('-')[0])}"
                first_prod_year = str(first_prod_year)
            Prism_data.loc[iwell, 'first_prod_year'] = first_prod_year
        Prism_data_dict[f'{iyear}'] = Prism_data

        # Combine into one array with common column names, replace nans with zeros, and sum annual production
        Enverus_data = pd.concat([DI_data, Prism_data], ignore_index=True)
        Enverus_data.loc[:, Enverus_data.columns.str.contains('GASPROD_')] = (
            Enverus_data.loc[:, Enverus_data.columns.str.contains('GASPROD_')].fillna(0)
        )
        Enverus_data.loc[:, Enverus_data.columns.str.contains('OILPROD_')] = (
            Enverus_data.loc[:, Enverus_data.columns.str.contains('OILPROD_')].fillna(0)
        )
        Enverus_data.loc[:, Enverus_data.columns.str.contains('WATERPROD_')] = (
            Enverus_data.loc[:, Enverus_data.columns.str.contains('WATERPROD_')].fillna(0)
        )

        # Calculate cummulative annual production totals for Gas, Oil, Water
        Enverus_data['CUM_GAS'] = (
            Enverus_data.loc[:, Enverus_data.columns.str.contains('GASPROD_')].sum(1)
        )
        Enverus_data['CUM_OIL'] = (
            Enverus_data.loc[:, Enverus_data.columns.str.contains('OILPROD_')].sum(1)
        )
        Enverus_data['CUM_WATER'] = (
            Enverus_data.loc[:, Enverus_data.columns.str.contains('WATERPROD_')].sum(1)
        )

        # Save out the data for that year
        Enverus_data_dict[f'{iyear}'] = Enverus_data

        del Prism_data
        del DI_data  # save memory space 

    # Correct Enverus Data for Select States

    # 1) Read In Coverage Table from State Well Counts File from ERG
    # (specifies the first year with bad data and which years need to be corrected;
    # all years including and after the first bad year of data need to be corrected)

    ERG_StateWellCounts_LastGoodDataYear = (
        pd.read_excel(
            enverus_well_counts_path,
            sheet_name="2021 - Coverage",
            usecols={"State", "Last Good Year"},
            skiprows=2,
            nrows=40)
        )

    # 2) Loops through the each state and year in Enverus to determine if the data for that particualar year needs to 
    # be corrected. At the moment, the only corrections ERG makes to the data is to use the prior year of data if there
    # is no new Enverus data reportd for that state. If a particular state is not included for any years in the Enverus
    # dataset, then a row of zeros is added to the Enverus table for that year.

    for istate in np.arange(0, len(state_gdf)):
        correctdata = 0
        istate_code = state_gdf['state_code'][istate]
        lastgoodyear = ERG_StateWellCounts_LastGoodDataYear['Last Good Year'][ERG_StateWellCounts_LastGoodDataYear['State'] == istate_code].values
        if lastgoodyear == max_year:
            # if state isn't included in correction list, don't correct any data
            lastgoodyear = max_year+5

        for iyear in years:
            enverus_data_temp = Enverus_data_dict[f'{iyear}'].copy()
            state_list = np.unique(enverus_data_temp['STATE_CODE'])
            if istate_code in state_list:
                inlist = 1
            else:
                inlist = 0
            # if the state is included in Enverus data, or had data for at least one good year
            if inlist == 1 or correctdata == 1:
                # if first year, correctdata will be zero, but inlist will also be zero if no Enverus data
                # check to see whether corrections are necessary for the given year/state
                if iyear == (lastgoodyear):
                    print(istate_code, iyear, 'last good year')
                    # This is the last year of good data. Do not correct the data but save
                    # but so that this data can be used for all following years for that state
                    temp_data = enverus_data_temp[enverus_data_temp['STATE_CODE'] == istate_code]
                    correctdata = 1
                elif iyear > lastgoodyear:
                    print(istate_code, iyear)
                    # correct data for all years equal to and after the first bad year (remove old data first if necessary)
                    if inlist == 1:
                        enverus_data_temp = enverus_data_temp[enverus_data_temp['STATE_CODE'] != istate_code]
                    enverus_data_temp = pd.concat([enverus_data_temp, temp_data], ignore_index=True)
                    print(istate_code + ' data for ' + str(iyear) + ' were corrected with ' + str(lastgoodyear) + ' data')
                else:
                    no_corrections = 1

            if inlist == 0 and correctdata == 0:
                # if there is no Enverus data for a given state, and there was no good data, add a row with default values
                print(istate_code + ' has no Enverus data in the year ' + str(iyear))

            # save that year of Enverus data
            enverus_data_temp.reset_index(drop=True, inplace=True)
            Enverus_data_dict[f'{iyear}'] = enverus_data_temp.copy()
            tempoutput_filename = f'formatted_raw_enverus_tempoutput_{iyear}.csv'
            tempoutput_filepath = os.path.join(intermediate_outputs_path, tempoutput_filename)
            enverus_data_temp.to_csv(tempoutput_filepath, index=False)

    return None
