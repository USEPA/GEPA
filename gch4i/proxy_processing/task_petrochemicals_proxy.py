# %%
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile
import calendar
import datetime

from pyarrow import parquet
import pandas as pd
import osgeo
import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
)

from gch4i.utils import name_formatter


@mark.persist
@task(id="petrochemicals_proxy")
def task_get_petrochemicals_proxy_data(
    inventory_workbook_path: Path = ghgi_data_dir_path / "2B8_petrochemicals/State_Petrochemicals_1990-2022.xlsx",
    subpart_x_path = "https://data.epa.gov/efservice/x_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "petrochemicals_proxy.parquet",
):
    """
    Four petrochemical production facilities with Arylonitrile production report
    for the 2012-2022 reporting period.

    Location information (e.g., latitude/longitude) for each petrochemical facility
    is taken from the Subpart X database. Initially, only two facilities were matched
    to a location. After looking up the remaining facilities on Google Maps, 
    it was determined that the remaining two facilities had mistakes in their city 
    and/or facility name to allow matching locations to Subpart X.

    The Cornerstone facility in the GHGI workbook had its city updated to Waggaman, LA
    instead of Avondale, LA.

    The Texas INEOS facility in the GHGI workbook had its facility name updated to
    Green Lake Plant instead of INEOS and its city updated to Port Lavaca, TX instead
    of Green Lake, TX.

    Location information (e.g., latitude/longitude) for each petrochemical facility
    is taken from the Subpart X database.
    """

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

    # Get and format GHGI facility capacities
    ghgi_facilities_df = (
        pd.read_excel(
            inventory_workbook_path,
            sheet_name="SRI&ICIS",
            skiprows=1,
            nrows=130,
            usecols="A:D,AB:AL",
        )
        .rename(columns=lambda x: str(x).lower())
        .rename(columns={"company": "facility_name", "state": "state_name"})
        .rename(columns={"capacity.22": "2012", "capacity.23": "2013", "capacity.24": "2014",
        "capacity.25": "2015", "capacity.26": "2016", "capacity.27": "2017", 
        "capacity.28": "2018", "capacity.29": "2019", "capacity.30": "2020",
        "capacity.31": "2021", "capacity.32": "2022"})
        .query("petrochemical == 'Acrylonitrile'")
        .drop(columns=["petrochemical"])
        .astype({"state_name":str})
        .merge(state_gdf[["state_code", "state_name"]], on="state_name")
        # drop facilities that have all nan values
        .dropna(subset=[str(x) for x in list(range(2012,2023))])
        .reset_index(drop=True)
    )

    # Format facility names to drop previous facility names found in parentheses
    for ifacility in np.arange(0,len(ghgi_facilities_df)):
        facility_name = ghgi_facilities_df.loc[ifacility, 'facility_name']
        city = ghgi_facilities_df.loc[ifacility, 'city']
        shortened_facility_name = facility_name.partition(" (")[0]
        ghgi_facilities_df.loc[ifacility, "facility_name"] = shortened_facility_name.lower()
        ghgi_facilities_df.loc[ifacility, "city"] = city.lower()
    
    # Correct incorrect GHGI facility names and/or cities
    # Facilities were manually found on Google Maps to determine the correct city.
    # Facility names were changed to match the facility names found in the 
    # Addresses by Street and Addresses by Lat_Long tabs of the inventory workbook 
    # which were derived from the Subpart X datasets.

    # Cornerstone in Avondale, LA should have its city updated to Waggaman, LA.
    imatch = np.where((ghgi_facilities_df['facility_name'] == 'cornerstone') 
                       & (ghgi_facilities_df['city'] == 'avondale'))[0]
    ghgi_facilities_df.loc[imatch, 'city'] = 'waggaman'

    # INEOS in Green Lake, TX should have its facility name and city updated to 
    # Green Lake Plant in Port Lavaca, TX
    imatch = np.where((ghgi_facilities_df['facility_name'] == 'ineos') 
                       & (ghgi_facilities_df['city'] == 'green lake'))[0]
    ghgi_facilities_df.loc[imatch, 'facility_name'] = 'green lake plant'
    ghgi_facilities_df.loc[imatch, 'city'] = 'port lavaca'

    # Get and format Subpart X facility locations
    facility_locations_df = (
        pd.read_csv(
            subpart_x_path,
            usecols=("facility_name",
                     "facility_id",
                     "latitude",
                     "longitude",
                     "city",
                     "state")
            )
        .rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        .drop_duplicates(subset=['facility_id', 'city'], keep='first')
        .reset_index(drop=True)
    )

    # Match GHGI facilities to Subpart X facility locations
    for ifacility in np.arange(0,len(ghgi_facilities_df)):
        state_code_temp = ghgi_facilities_df.loc[ifacility, 'state_code']
        imatch = np.where((facility_locations_df['facility_name'].str.contains(ghgi_facilities_df.loc[ifacility,'facility_name'], case=False))\
                & (facility_locations_df['state_code']==state_code_temp))[0]
        ghgi_facilities_df.loc[ifacility, 'lat'] = facility_locations_df.loc[imatch[0], 'latitude']
        ghgi_facilities_df.loc[ifacility, 'lon'] = facility_locations_df.loc[imatch[0], 'longitude']

    # Format proxy data to consolidate years into a single column
    ghgi_facilities_w_locations_df = ghgi_facilities_df.melt(id_vars=[
        'facility_name', 'state_code', 'lat', 'lon'],
        value_vars=list(ghgi_facilities_df.columns.values)[3:14],
        var_name='year', value_name='capacity_kt')
    
    ghgi_facilities_w_locations_df['rel_emi'] = ghgi_facilities_w_locations_df.groupby(["state_code", "year"])['capacity_kt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    ghgi_facilities_w_locations_df = ghgi_facilities_w_locations_df.drop(columns='capacity_kt')
    
    # Create proxy gdf
    proxy_gdf = (
        gpd.GeoDataFrame(
            ghgi_facilities_w_locations_df,
            geometry=gpd.points_from_xy(
                ghgi_facilities_w_locations_df["lon"],
                ghgi_facilities_w_locations_df["lat"],
                crs=4326,
                ),
            )
        .drop(columns=["lat", "lon"])
        .astype({"year": int})
        .loc[:, ["year", "facility_name", "state_code", "geometry", "rel_emi"]]
        )

    proxy_gdf.to_parquet(output_path)
    return None

# %%
