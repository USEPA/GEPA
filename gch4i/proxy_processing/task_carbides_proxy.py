"""
Name:                   task_carbides_proxy.py
Date Last Modified:     2024-11-26
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of stationary combustion proxy emissions
Input Files:            - Proxy_Facilities.xlsx
                        - Carbide_Facilities_2012-2022.csv
Output Files:           - carbides_proxy.parquet
Notes:                  - V2 only had location of one facilitiy, in IL
                        - Documentation reflects method for locating other facilities
                        - Proxy_Facilities.xlsx is a manual entry of facility locations
                        gathered from correspondence with EPA.
"""
########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd


import geopandas as gpd

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    max_year,
    min_year
)

########################################################################################
# %% Load Path Files
GEPA_Carbide_Path = V3_DATA_PATH.parent / "GEPA_Source_Code" / "GEPA_Carbide"

EPA_Proxy_Facilities_Path = GEPA_Carbide_Path / "InputData/Proxy_Facilities.xlsx"

EPA_GHGRP_facilities_path = GEPA_Carbide_Path / "InputData/GHGRP/Carbide_Facilities_2012-2022.csv"

########################################################################################
# %% Pytask


@mark.persist
@task(id="carbides_proxy")
def task_get_carbides_proxy_data(
    EPA_Proxy_Facilities_Path: Path = EPA_Proxy_Facilities_Path,
    EPA_GHGRP_facilities_path: Path = EPA_GHGRP_facilities_path,
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "carbides_proxy.parquet"
):
    """
    There are two carbide production facilities in the 2012-2022 reporting period. The
    IL and KY facilities were identified via correspondence with EPA and stored in the
    Proxy_Facilities.xlsx file. The lat/lon were gathered from the
    Carbide_Facilities_2012-2022.csv; however, this script grabs the lat/lon directly
    from the GHGRP facility locations file, as opposed to manually entering them from
    the Proxy_Facilities.xlsx file.

    Alternative locations for the KY facility were identified via querying the
    Enforcement adn Compliance History Online (ECHO) database, filtering for facilities
    in Kentucky that use the 32910 NAICS Code. Three alternate facilities were
    identified:

    1. ABRAPOWER INC; 8055 DIXIE HWY; FLORENCE, KY
    2. ALLISON ABRASIVES INC; 141 INDUSTRY RD; LANCASTER, KY
    3. STEIN INC; 69 ARMCO RD; ASHLAND, KY

    These facilities are recorded here, but not used in the analysis.
    """

    # Proxy Facilities obtained from EPA correspondence
    proxy_facilities = (
        pd.read_excel(
            EPA_Proxy_Facilities_Path,
            sheet_name="Carbide_Facilities",
            usecols="A:D"  # Bring in Lat/Lon from facility locations (not manual entry)
        )
    )

    # GHGRP Facility Locations
    facility_locations_df = (
        pd.read_csv(
            EPA_GHGRP_facilities_path,
            usecols=("facility_name",
                     "facility_id",
                     "latitude",
                     "longitude",
                     "city",
                     "state",
                     "year"))
        .rename(columns={"state": "state_code"})
        .drop_duplicates(subset=['facility_id', 'city', 'year'], keep='first')
        .reset_index(drop=True)
        )

    proxy_df = (
        pd.merge(
            proxy_facilities,
            facility_locations_df[['year',
                                   'facility_id',
                                   'facility_name',
                                   'latitude',
                                   'longitude']],
            on=["facility_id", 'facility_name'],
            how="left"
        )
        .query(f"year >= {min_year} & year <= {max_year}")
    )

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
        .loc[:, ["facility_name", "state_code", "geometry", "year"]]
        .sort_values(by=["facility_name", "year"])
        .reset_index(drop=True)
    )

    proxy_gdf.to_parquet(output_path)
    return None
