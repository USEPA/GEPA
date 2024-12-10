# %%
from pathlib import Path
from typing import Annotated

from pyarrow import parquet  # noqa
import osgeo  # noqa
import duckdb
import geopandas as gpd
import pandas as pd
from geopy.geocoders import Nominatim
from pytask import Product, mark, task

from gch4i.config import (  # noqa
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    proxy_data_dir_path,
)
from gch4i.utils import name_formatter

# gpd.options.io_engine = "pyogrio"


# %%
@mark.persist
@task(id="ferro_proxy")
def task_ferro_proxy_data(
    EPA_inventory_path: Path = list(
        ghgi_data_dir_path.rglob("State_Ferroalloys_1990-2022.xlsx")
    )[0],
    frs_path: Path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV",
    subpart_k_url: str = (
        "https://data.epa.gov/efservice/k_subpart_level_information/"
        "pub_dim_facility/ghg_name/=/Methane/CSV"
    ),
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    dst_path: Annotated[Path, Product] = proxy_data_dir_path / "ferro_proxy.parquet",
) -> None:

    # The facilities have multiple reporting units for each year. This will read in the
    # facilities data and compute the facility level sum of emissions for each
    # year. This pulls from the raw table but ends in the same form as the table on
    # sheet "GHGRP_kt_Totals"

    # read in the SUMMARY facilities emissions data
    # %%
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

    ferro_facilities_df = (
        pd.read_excel(
            EPA_inventory_path,
            sheet_name="Ferroalloy Calculations",
            skiprows=72,
            nrows=12,
            usecols="A:AJ",
        )
        .rename(columns=lambda x: str(x).lower())
        .rename(columns={"facility": "facility_name", "state": "state_name"})
        .drop(columns=["year opened"])
        .melt(
            id_vars=["facility_name", "state_name"],
            var_name="year",
            value_name="est_ch4",
        )
        .astype({"year": int})
        .assign(formatted_fac_name=lambda df: name_formatter(df["facility_name"]))
        .merge(state_gdf[["state_code", "state_name"]], on="state_name")
        .query("year.between(@min_year, @max_year)")
    )

    # STEP 3.1: Get and format FRS and Subpart K facility locations data
    # this will read in both datasets, format the columns for the facility name, 2
    # letter state code, latitude and longitude. Then merge the two tables together,
    # and create a formatted facility name that eases matching of our needed facilities
    # to the location data.

    frs_raw = duckdb.execute(
        (
            "SELECT primary_name, state_code, latitude83, longitude83 "
            f"FROM '{frs_path}' "
        )
    ).df()

    subpart_k_raw = pd.read_csv(subpart_k_url)

    frs_df = (
        frs_raw.rename(columns=str.lower)
        .drop_duplicates(subset=["primary_name", "state_code"])
        .rename(columns={"primary_name": "facility_name"})
        .rename(columns=lambda x: x.strip("83"))
        .dropna(subset=["latitude", "longitude"])
    )

    subpart_k_df = (
        subpart_k_raw.drop_duplicates(subset=["facility_name", "state"])
        .rename(columns={"state": "state_code"})
        .loc[
            :,
            ["facility_name", "state_code", "latitude", "longitude"],
        ]
        .dropna(subset=["latitude", "longitude"])
    )

    # merge the two datasets together, format the facility name, drop any duplicates.
    facility_locations = (
        pd.concat([subpart_k_df, frs_df], ignore_index=True)
        .assign(formatted_fac_name=lambda df: name_formatter(df["facility_name"]))
        .drop_duplicates(subset=["formatted_fac_name", "state_code"])
    )

    # create a table of unique facility names to match against the facility locations
    # data.
    unique_facilities = (
        ferro_facilities_df[["formatted_fac_name", "state_code"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # try to match facilities based on the formatted name and state code
    matching_facilities = unique_facilities.merge(
        facility_locations, on=["formatted_fac_name", "state_code"], how="left"
    )

    # NOTE: We are missing 4 facilities. Manually, we checked the 4 missing facilities
    # for partial name matches in the subpart/frs facility data. We found 3 facilities.
    # 1 was still missing. Using partial matching of name and state, we assign the
    # coords to our records.

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "bear metallurgical co",
        ["facility_name", "latitude", "longitude"],
    ] = (
        facility_locations[
            facility_locations["formatted_fac_name"].str.contains("bear metal")
            & facility_locations["state_code"].eq("PA")
        ]
        .loc[:, ["facility_name", "latitude", "longitude"]]
        .values
    )

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "stratcor inc",
        ["facility_name", "latitude", "longitude"],
    ] = (
        facility_locations[
            facility_locations["formatted_fac_name"].str.contains("stratcor")
            & facility_locations["state_code"].eq("AR")
        ]
        .loc[:, ["facility_name", "latitude", "longitude"]]
        .values
    )

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "metallurg vanadium corp",
        ["facility_name", "latitude", "longitude"],
    ] = (
        facility_locations[
            facility_locations["formatted_fac_name"].str.contains("vanadium inc")
            & facility_locations["state_code"].eq("OH")
        ]
        .loc[:, ["facility_name", "latitude", "longitude"]]
        .values
    )

    # NOTE: The final facility, thompson creek, has no name match in the facilities
    # data, so we use the city and state provided in the inventory data to get its
    # location.
    # https://en.wikipedia.org/wiki/Thompson_Creek_Metals.
    # thompson creek metals co inc was acquired in 2016 by Centerra Gold.
    # A search for Centerra in PA turns up 2 records in the combine subpart K and FRS
    # dataset. But Centerra Co-op is an agricultural firm, not a ferroalloy facility.
    # https://www.centerracoop.com/locations
    # display(
    #     facility_locations[
    #         facility_locations["formatted_fac_name"].str.contains("centerra")
    #         & facility_locations["state_code"].eq("PA")
    #     ]
    # )

    fac_locations = pd.read_excel(
        EPA_inventory_path,
        sheet_name="USGS_2008_Facilities",
        skiprows=4,
        nrows=12,
        usecols="A:C",
    ).drop(columns="Unnamed: 1")
    t_creek_city_state = fac_locations.loc[
        fac_locations["Company"] == "Thompson Creek Metals Co.", "Plant location"
    ].values[0]

    geolocator = Nominatim(user_agent="RTI testing")
    t_creek_loc = geolocator.geocode(t_creek_city_state)

    matching_facilities.loc[
        matching_facilities["formatted_fac_name"] == "thompson creek metals co inc",
        ["latitude", "longitude"],
    ] = [t_creek_loc.latitude, t_creek_loc.longitude]

    # we now have all facilities with addresses.
    ferro_facilities_gdf = ferro_facilities_df.merge(
        matching_facilities.drop(columns="facility_name"),
        on=["formatted_fac_name", "state_code"],
        how="left",
    )
    ferro_facilities_gdf = gpd.GeoDataFrame(
        ferro_facilities_gdf.drop(columns=["latitude", "longitude"]),
        geometry=gpd.points_from_xy(
            ferro_facilities_gdf["longitude"],
            ferro_facilities_gdf["latitude"],
            crs=4326,
        ),
    )

    # make sure the merge gave us the number of results we expected.
    if not (ferro_facilities_gdf.shape[0] == ferro_facilities_df.shape[0]):
        print("WARNING the merge shape does not match the original data")
    ferro_facilities_gdf
    # # save a shapefile of the v3 ferro facilities for reference
    # fac_locations = ferro_facilities_gdf.dissolve("formatted_fac_name")
    # fac_locations[fac_locations.is_valid].loc[:, ["geometry"]].to_file(
    #     tmp_data_dir_path / "v3_ferro_facilities.shp.zip", driver="ESRI Shapefile"
    # )
    ferro_facilities_gdf.to_parquet(dst_path)
    # %%
