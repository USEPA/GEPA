"""
Name:                   ng_oil_production_utils.py
Date Last Modified:     2025-01-30
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Generates functions used in the processing of natural gas and
                            petroleum production proxy data.
Input Files:            -
Output Files:           -
"""

# %% Import Libraries
from pathlib import Path
import os

import pandas as pd

import geopandas as gpd
import numpy as np


from gch4i.config import (
    global_data_dir_path,
    sector_data_dir_path
)

# File Paths
state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"
enverus_production_path: Path = sector_data_dir_path / "enverus/production"
intermediate_outputs_path: Path = enverus_production_path / "intermediate_outputs"
nei_path: Path = sector_data_dir_path / "nei_og"

# State ANSI data
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


# Function to calculate relative emissions for Enverus data
def calc_enverus_rel_emi(df):
    df['rel_emi'] = (
        df.groupby(["state_code", "year"])['proxy_data']
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    )
    df = df.drop(columns='proxy_data')
    return df


# function to format proxy data into geodataframes
def enverus_df_to_gdf(df):
    gdf = (
        gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df["longitude"],
                df["latitude"],
                crs=4326
            )
        )
        .drop(columns=["latitude", "longitude"])
        .loc[:, ["year", "year_month", "state_code", "rel_emi", "geometry"]]
    )
    return gdf


# NEI FIPS codes
fips_codes_df = pd.DataFrame({'state_code': ['IL', 'IN', 'KS', 'OK', 'PA', 'WV'],
                              'fips_code': ['17', '18', '20', '40', '42', '54']})

# NEI data year assignments
# All years use the data affiliated with their year except the following exceptions:
    # 2012: use 2011 data
    # 2013: use 2014 data
    # 2015: use 2014 data
    # 2016: use 2017 data
nei_data_years = pd.DataFrame(
    {'year': [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
     'nei_data': [2011, 2014, 2014, 2014, 2017, 2017, 2018, 2019, 2020, 2021, 2022]
     })

# NEI text file and shapefile names:
# Natural Gas Well Counts
ng_well_count_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_698_NOFILL.txt', 'USA_698_NOFILL.txt', 'USA_698_NOFILL.txt',
                  'GAS_WELLS', 'GAS_WELLS', 'GAS_WELL', '_698', 'GasWells'],
    })
# Natural Gas Well-Level Production Volumes
ng_gas_prod_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_696_NOFILL.txt', 'USA_696_NOFILL.txt', 'USA_696_NOFILL.txt',
                  'GAS_PRODUCTION', 'GAS_PRODUCTION', 'GAS_PRODUCTION', '_696',
                  'GasProduction'],
    })
# Natural Gas Water Production Volumes
ng_water_prod_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_6832_NOFILL.txt', 'USA_6832_NOFILL.txt', 'USA_6832_NOFILL.txt',
                  'PRODUCED_WATER_GAS', 'PRODUCED_WATER_GAS', 'PRODUCED_WATER_GAS',
                  '_6832', 'ProducedWaterGasWells'],
    })
# Natural Gas Well Completions
ng_comp_count_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_678_NOFILL.txt', 'USA_678_NOFILL.txt', 'USA_678_NOFILL.txt',
                  'COMPLETIONS_GAS', 'COMPLETIONS_GAS', 'COMPLETIONS_GAS', '_678',
                  'GasWellCompletions'],
    })
# Natural Gas Drilled Gas Wells
ng_spud_count_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_671_NOFILL.txt', 'USA_671_NOFILL.txt', 'USA_671_NOFILL.txt',
                  'SPUD_GAS', 'SPUD_GAS', 'SPUD_GAS', '_671', 'SpudCountGasWells'],
    })
# Oil Well Counts
oil_well_count_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_695_NOFILL.txt', 'USA_695_NOFILL.txt', 'USA_695_NOFILL.txt',
                  'OIL_WELLS', 'OIL_WELLS', 'OIL_WELL', '_695', 'OILWells'],
    })
# Oil Well-Level Production Volumes
oil_oil_prod_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_694_NOFILL.txt', 'USA_694_NOFILL.txt', 'USA_694_NOFILL.txt',
                  'OIL_PRODUCTION', 'OIL_PRODUCTION', 'OIL_PRODUCTION', '_694',
                  'OilProduction'],
    })
# Oil Water Production Volumes
oil_water_prod_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_6833_NOFILL.txt', 'USA_6833_NOFILL.txt', 'USA_6833_NOFILL.txt',
                  'PRODUCED_WATER_OIL', 'PRODUCED_WATER_OIL', 'PRODUCED_WATER_OIL',
                  '_6833', 'ProducedWaterOilWells'],
    })
# Oil Well Completions
oil_comp_count_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_685_NOFILL.txt', 'USA_685_NOFILL.txt', 'USA_685_NOFILL.txt',
                  'COMPLETIONS_OIL', 'COMPLETIONS_OIL', 'COMPLETIONS_OIL', '_685',
                  'OilWellCompletions'],
    })
# Oil Drilled Gas Wells
oil_spud_count_file_names = pd.DataFrame({
    'data_year': [2011, 2014, 2017, 2018, 2019, 2020, 2021, 2022],
    'file_name': ['USA_681_NOFILL.txt', 'USA_681_NOFILL.txt', 'USA_681_NOFILL.txt',
                  'SPUD_OIL', 'SPUD_OIL', 'SPUD_OIL', '_681', 'SpudCountOilWells'],
    })


# Function to get the specific file name for a given year
def get_nei_file_name(nei_data_year, nei_file_names):
    nei_file_name = (
        nei_file_names[nei_file_names['data_year'] == nei_data_year]['file_name'].values[0]
    )
    return nei_file_name


# Function to get raw NEI textfile and shapefile data for the specific proxy of interest
def get_raw_NEI_data(ghgi_year, data_year, file_name):
    if data_year <= 2017:
        # NEI textfile data (data_year <= 2017) (2011, 2014, 2016, 2017)
        nei_textfile_name = f"CONUS_SA_FILES_{data_year}/{file_name}"
        nei_textfile_path = os.path.join(nei_path, nei_textfile_name)
        data_temp = pd.read_csv(nei_textfile_path, sep='\t', skiprows=25)
        data_temp = data_temp.drop(["!"], axis=1)
        data_temp.columns = ['Code', 'FIPS', 'COL', 'ROW', 'Frac', 'Abs', 'FIPS_Total',
                             'FIPS_Running_Sum']
        data_temp = data_temp.astype({"FIPS": str})
        # if water production data (gas: 6832, oil: 6833)
        if file_name == 'USA_6832_NOFILL.txt' or file_name == 'USA_6833_NOFILL.txt':
            if data_year < 2016:
                data_temp = (
                    data_temp
                    # query states: IL, IN, KS, OK, PA, WV
                    .query("FIPS.str.startswith('17') | FIPS.str.startswith('18') | FIPS.str.startswith('20') | FIPS.str.startswith('40') | FIPS.str.startswith('42') | FIPS.str.startswith('54')")
                    .reset_index(drop=True)
                            )
                colmax = data_temp['COL'].max()
                colmin = data_temp['COL'].min()
                rowmax = data_temp['ROW'].max()
                rowmin = data_temp['ROW'].min()
            else:
                data_temp = (
                    data_temp
                    # query states: IL, IN, KS, OK, PA
                    .query("FIPS.str.startswith('17') | FIPS.str.startswith('18') | FIPS.str.startswith('20') | FIPS.str.startswith('40') | FIPS.str.startswith('42')")
                    .reset_index(drop=True)
                            )
                colmax = data_temp['COL'].max()
                colmin = data_temp['COL'].min()
                rowmax = data_temp['ROW'].max()
                rowmin = data_temp['ROW'].min()
        # non-water production proxies (IL, IN)
        else:
            data_temp = (
                data_temp
                # query states: IL, IN
                .query("FIPS.str.startswith('17') | FIPS.str.startswith('18')")
                .reset_index(drop=True)
                        )
            colmax = data_temp['COL'].max()
            colmin = data_temp['COL'].min()
            rowmax = data_temp['ROW'].max()
            rowmin = data_temp['ROW'].min()
        # NEI reference grid shapefile with lat/lon locations
        nei_reference_grid_path = os.path.join(nei_path, "NEI_Reference_Grid_LCC_to_WGS84_latlon.shp")
        nei_reference_grid = (gpd.read_file(nei_reference_grid_path).to_crs(4326))
        nei_reference_grid = (
            nei_reference_grid
            .assign(cellid_column=nei_reference_grid.cellid.astype(str).str[0:4].astype(int))
            .assign(cellid_row=nei_reference_grid.cellid.astype(str).str[5:].astype(int))
            .query(f"cellid_column <= {colmax} & cellid_column >= {colmin}")
            .query(f"cellid_row <= {rowmax} & cellid_row >= {rowmin}")
            .reset_index(drop=True)
                            )
        # Match lat/lon locations from reference grid to nei data
        for idx in np.arange(0, len(data_temp)):
            # Add in lat/lon
            icol = data_temp['COL'][idx]
            irow = data_temp['ROW'][idx]
            match = np.where((icol == nei_reference_grid.loc[:, 'cellid_column']) &
                             (irow == nei_reference_grid.loc[:, 'cellid_row']))[0][0]
            match = int(match)
            # data_temp.loc[idx,'Lat'] = nei_reference_grid.loc[match, 'Latitude']
            # data_temp.loc[idx,'Lon'] = nei_reference_grid.loc[match, 'Longitude']
            data_temp.loc[idx, 'geometry'] = nei_reference_grid.loc[match, 'geometry']
            # Add in state_code
            ifips = data_temp.loc[idx, 'FIPS'][0:2]
            data_temp.loc[idx, 'state_code'] = (
                fips_codes_df.loc[np.where(ifips == fips_codes_df.loc[:, 'fips_code'])[0][0], 'state_code']
            )
        data_temp = data_temp[['state_code', 'Abs', 'geometry']]
        data_temp = data_temp.rename(columns={'Abs': 'activity_data'})

    else:
        # NEI shapefile data (data_year > 2017) (2018, 2019, 2021, 2022)
        state_geometries = state_gdf[["state_code","geometry"]]
        nei_file_name = f"CONUS_SA_FILES_{data_year}"
        nei_file_path = os.path.join(nei_path, nei_file_name)
        data_temp = gpd.read_file(nei_file_path, layer=file_name)
        data_temp = data_temp.to_crs(4326)
        data_temp = gpd.tools.sjoin(data_temp, state_gdf, how="left")

        # water production data (IL, IN, KS, OK, PA)
        if file_name == 'PRODUCED_WATER_GAS' or file_name == '_6832' or file_name == 'ProducedWaterGasWells':
            states_to_query = ['IL', 'IN', 'KS', 'OK', 'PA']
        # non-water production proxies (IL, IN)
        else:
            states_to_query = ['IL', 'IN']

        # query relevant states
        data_temp = data_temp.query('state_code.isin(@states_to_query)')

        # grab activity data depending on column name (changes by year)
        if data_year == 2018 or data_year == 2019 or data_year == 2020:
            data_temp = data_temp[['state_code', 'ACTIVITY', 'geometry']]
            data_temp = data_temp.rename(columns={'ACTIVITY': 'activity_data'})
        if data_year == 2021:
            data_temp = data_temp[['state_code', 'GRID_AC', 'geometry']]
            data_temp = data_temp.rename(columns={'GRID_AC': 'activity_data'})
        if data_year == 2022:
            data_temp = data_temp[['state_code', 'GRID_ACTIV', 'geometry']]
            data_temp = data_temp.rename(columns={'GRID_ACTIV': 'activity_data'})

    # convert activity data to relative emissions (idata / sum(state data))
    data_temp['rel_emi'] = (
        data_temp.groupby(["state_code"])['activity_data']
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    )
    monthly_data_temp = data_temp.copy()
    monthly_data_temp['rel_emi'] = monthly_data_temp['rel_emi'] * 1/12
    monthly_data_temp = monthly_data_temp.drop(columns='activity_data')

    # convert proxy data to monthly (assume 1/12 of annual proxy is assigned to each month)
    nei_proxy_data = pd.DataFrame()
    for imonth in range(1, 13):
        imonth_str = f"{imonth:02}"  # convert to 2-digit months
        data_temp_imonth = monthly_data_temp.copy()
        data_temp_imonth = data_temp_imonth.assign(year_month=str(ghgi_year)+'-'+imonth_str)
        nei_proxy_data = pd.concat([nei_proxy_data, data_temp_imonth])
    nei_proxy_data = nei_proxy_data.assign(year=ghgi_year)
    nei_proxy_data = (
        nei_proxy_data[['year', 'year_month', 'state_code', 'rel_emi', 'geometry']]
        .reset_index(drop=True)
        )
    return nei_proxy_data
