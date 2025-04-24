"""
Name:                   task_iron_steel_proxy.py
Date Last Modified:     2025-02-05
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of stationary combustion proxy emissions
Input Files:            Subpart Q Iron API Call:
                        - https://data.epa.gov/efservice/q_subpart_level_information/
                            pub_dim_facility/ghg_name/=/Methane/CSV
                        Inventory: {ghgi_data_dir_path}/
                            2C1_iron_and_steel/State_Iron-Steel_1990-2022.xlsx
                        State Geo: {global_data_dir_path}/tl_2020_us_state.zip
Output Files:           - {proxy_data_dir_path}/iron_steel_proxy.parquet
"""
########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
# import os
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
# import numpy as np

import geopandas as gpd

from gch4i.config import (
    # V3_DATA_PATH,
    emi_data_dir_path,
    proxy_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path
)

########################################################################################
# %% Pytask


@mark.persist
@task(id="iron_steel_proxy")
def task_get_iron_steel_proxy_data(
    inventory_workbook_path: Path = (
        ghgi_data_dir_path / "2C1_iron_and_steel/State_Iron-Steel_1990-2022.xlsx"
    ),
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    subpart_q_path="https://data.epa.gov/efservice/q_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV",
    emi_iron_steel_path: Path = emi_data_dir_path / "iron_steel_emi.csv",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "iron_steel_proxy.parquet"
):
    """
    GHGRP Facilities used for proxies were pulled from the inventory workbook
    [GHGRP_Facilities] tab. "FACILIY_ID" did not match GHGRP IDs, so direct matching
    latitude, longitude, and ghg_quantity was not possible. Instead, GHGRP facility
    information was accessed through the EPA API.

    THE subpart Q data, accessed through the API, was used to identify the GHGRP
    facility_id that correspond with the inventory workbook's proxy locations. The
    facility_id was then used to access the ghg_quantity, latitude, and longitude data.

    Several facilities have multiple names/addresses associated with the same
    facility_id. Direct matches were made to the inventory workbook proxy locations to
    maintain consistency. Direct inputation was used to correct missingness and/or
    clarify multiple addresses.
    """

    # Get and format GHGI facilities data
    ghgi_facilities_df = (
        pd.read_excel(
            inventory_workbook_path,
            sheet_name="GHGRP_Facilities",
            skiprows=2,
            nrows=132,
            usecols="A:J",
        )
        .rename(columns=lambda x: str(x).lower())
    )

    # Clean GHGI facilities data
    ghgi_facilities_df = (
        ghgi_facilities_df.rename(columns={"state": "state_code",
                                           "years operated": "years_operated"})
        .drop(
            columns=["address2", "latitude", "longitude", "facility_id", "city", "zip"]
            )
        .reset_index(drop=True)
    )

    # Get full subpart Q data
    subpart_Q = (
        pd.read_csv(
            subpart_q_path,
            usecols=["facility_id",
                     "facility_name",
                     "reporting_year",
                     "ghg_quantity",
                     "latitude",
                     "longitude",
                     "address1"
                     ])
        .rename(columns={"reporting_year": "year"})
    )

    # Merge GHGI facilities with subpart Q facilities to grab facility_id
    ghgi_facilities_df = (
        pd.merge(ghgi_facilities_df[['facility_name', 'address1',
                                     'state_code', 'years_operated']],
                 subpart_Q[['facility_name', 'address1', 'facility_id']],
                 on=['facility_name', 'address1'],
                 how='left')
        .drop_duplicates()
    )
    # 6 missing facility_ids
    unmatched_df = ghgi_facilities_df[ghgi_facilities_df['facility_id'].isna()]
    # Merge on address alone to fill missing facility_ids
    fill_missing = (
        unmatched_df[['facility_name', 'address1']]
        .merge(
            subpart_Q[['address1', 'facility_id']].drop_duplicates(),
            on='address1',
            how='left')
            )

    # Update df with missing facility_ids
    ghgi_facilities_df = ghgi_facilities_df.merge(
        fill_missing,
        on='address1',
        how='left',
        suffixes=('_x', '_y')
        )
    ghgi_facilities_df = (
        ghgi_facilities_df.assign(
            facility_id=(
                ghgi_facilities_df['facility_id_y']
                .fillna(ghgi_facilities_df['facility_id_x'])
                )
        )
        .drop(columns=['facility_id_x', 'facility_id_y', 'facility_name_y'])
        .rename(columns={'facility_name_x': 'facility_name'})
    )

    # 2 remaining missing facilities:
    # 99	Kentucky Electric Steel Company	2704 S BIG RUN RD 	KY	2012
    # 130	Steel Dynamics Southwest, LLC	7575 West Jefferson Blvd	IN	2022

    # Steel Dynamics Southwest :: different address1 :: same lat/lon
    # Insert ID:: 1014686
    ghgi_facilities_df.loc[ghgi_facilities_df['facility_name'] == 'Steel Dynamics Southwest, LLC', 'facility_id'] = 1014686

    # Kentucky Electric Steel :: address issues
    # Insert ID:: 1008263
    ghgi_facilities_df.loc[ghgi_facilities_df['facility_name'] == 'Kentucky Electric Steel Company', 'facility_id'] = 1008263

    # Make facility_id an integer
    ghgi_facilities_df['facility_id'] = ghgi_facilities_df['facility_id'].astype(int)

    # Clean ghgi years operated
    def extract_years(entry):
        entry = str(entry)

        if ' ' in entry:
            entry = entry.replace(' ', '')
            entry = entry.replace('.', ',')
        return entry

    # Explode years operated
    def expand_years(entry):
        years = []
        for part in entry.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                years.extend(range(start, end + 1))
            else:
                years.append(int(part))
        return years

    ghgi_facilities_df['years_operated'] = (
        ghgi_facilities_df['years_operated'].apply(extract_years)
    )
    ghgi_facilities_df['year'] = (
        ghgi_facilities_df['years_operated'].apply(expand_years)
    )
    ghgi_facilities_df = (
        ghgi_facilities_df
        .explode('year')
        .drop(columns=['years_operated'])
        .reset_index(drop=True)
    )

    # Subset subpart Q data to unique facility_id and year
    subpart_Q_red = (
        subpart_Q.drop_duplicates(subset=['facility_id', 'year'], keep='first')
    )

    # Merge ghgi facility data with reduced subpart Q data to get ghg_quantity, lat, lon
    proxy_df = (
        pd.merge(
            ghgi_facilities_df[['facility_name', 'facility_id', 'year', 'state_code']],
            subpart_Q_red[['facility_id', 'year', 'ghg_quantity', 'latitude', 'longitude']],
            on=['facility_id', 'year'],
            how='left')
    )

    # ID: 1001699; Name: Liberty Steel Georgetown Holdings LLC; has no 2018 data. Input 0 for ghg_quantity
    # Liberty Steel Georgetown Holdings LLC :: missing 2018 data
    proxy_df.loc[(proxy_df['facility_id'] == 1001699) & (proxy_df['year'] == 2018), ['ghg_quantity', 'latitude', 'longitude']] = [0, 33.36792, -79.29486]

    ####################################################################################
    # %% Generate Relative Emissions

    # Normalize relative emissions to state and year for each facility

    # Step 0: Drop empty ghg_quantity
    proxy_df = proxy_df[proxy_df['ghg_quantity'] > 0]

    # Step 1: Calculate the sum of emissions for each state-year group
    proxy_df['total_emissions'] = (
        proxy_df.groupby(['state_code', 'year'])['ghg_quantity'].transform('sum')
    )

    # Step 2: Normalize the emissions for each facility
    proxy_df['rel_emi'] = proxy_df['ghg_quantity'] / proxy_df['total_emissions']

    # Step 3: Drop the total_emissions column
    proxy_df = proxy_df.drop(columns=['total_emissions'])

    ####################################################################################
    # %% Geopandas conversion

    proxy_gdf = (
        gpd.GeoDataFrame(
            proxy_df,
            geometry=gpd.points_from_xy(
                proxy_df["longitude"],
                proxy_df["latitude"],
                crs=4326,
            )
        )
        .drop(columns=["latitude", "longitude"])
        .loc[:, ["facility_id", "facility_name", "state_code", "year", "rel_emi", "geometry"]]
        .sort_values(by=["facility_id", "year"])
        .reset_index(drop=True)
    )

    ####################################################################################
    # %% Alternative Proxy Data

    """
    Missing proxy data for the following states:
    CA: 2020
    MN: 2012-2022
    TX: 2016-2022
    """

    # Read in Emissions Data
    emi_df = (
        pd.read_csv(emi_iron_steel_path)
        .query("ghgi_ch4_kt != 0")
    )

    # Read in State Geo Data
    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    # Get unique state code-year pairs without proxy data
    emi_states = set(emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    proxy_states = set(proxy_gdf[['state_code', 'year']].itertuples(index=False, name=None))

    missing_states = emi_states.difference(proxy_states)

    # Add missing states alternative data to grouped_proxy
    if missing_states:
        # Create alternative proxy from missing states
        alt_proxy = (
            pd.DataFrame(missing_states, columns=['state_code', 'year'])
            # Assign facility_id and _name to NA, and make rel_emi = 1
            .assign(
                facility_id=pd.NA,
                facility_name=pd.NA,
                rel_emi=1
            )
            # Merge state polygon geometry
            .merge(
                state_gdf[['state_code', 'geometry']],
                on='state_code',
                how='left'
            )
        )
        # Convert to GeoDataFrame
        alt_proxy = gpd.GeoDataFrame(alt_proxy, geometry='geometry', crs='EPSG:4326')
        # Append to grouped_proxy
        proxy_gdf = pd.concat([proxy_gdf, alt_proxy], ignore_index=True)

    proxy_gdf.to_parquet(output_path)
    return None
