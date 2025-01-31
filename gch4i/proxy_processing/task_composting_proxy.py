"""
Name:                   task_composting_proxy.py
Date Last Modified:     2025-01-30
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This script combined 4 different facility level datasets into
                        a single composting proxy. The datasets are from the FRS,
                        Biocycle, Excess Food, and the STA Certified Compost
                        Participants Map. The script removes duplicate facilities based
                        on a spatial tolerance and normalizes the emissions by state.
                        The script also adds a facility for DC since it has no
                        facilities listed but has reported emissions.
Input Files:            - CompostingFacilities.xlsx
                        - NATIONAL_FACILITY_FILE.CSV
                        - NATIONAL_NAICS_FILE.CSV
                        - biocycle_locs_clean.csv
                        - STA Certified Compost Participants Map.kml
                        - tl_2020_us_state.zip
Output Files:           - composting_proxy.parquet
"""

# %%
from pathlib import Path
from typing import Annotated

from pyarrow import parquet  # noqa
import osgeo  # noqa
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from pytask import Product, mark, task
from shapely import Point, wkb

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,  # noqa
    sector_data_dir_path,
)
from gch4i.utils import name_formatter, normalize

# gpd.options.io_engine = "pyogrio"


COMPOSTING_FRS_NAICS_CODE = 562219
DUPLICATION_TOLERANCE_M = 250

composting_dir = sector_data_dir_path / "composting"
# %%


@mark.persist
@task(id="composting_proxy")
def get_composting_proxy_data(
    excess_food_op_path: Path = composting_dir / "CompostingFacilities.xlsx",
    frs_facility_path: Path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV",
    frs_naics_path: Path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV",
    biocycle_path: Path = composting_dir / "biocycle_locs_clean.csv",
    comp_council_path: Path = (
        composting_dir / "STA Certified Compost Participants Map.kml"
    ),
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    dst_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "composting_proxy.parquet"
    ),
):
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

    excess_food_df = (
        pd.read_excel(
            excess_food_op_path,
            sheet_name="Data",
            usecols=["Name", "Latitude", "Longitude"],
        ).rename(columns=str.lower)
        # .rename(columns={"state": "state_code"})
        # .query("state_code.isin(@state_info_df['state_code'])")
        .assign(
            formatted_fac_name=lambda df: name_formatter(df["name"]), source="ex_food"
        )
    )
    excess_food_df

    # get the composting facilities from the FRS database based on the NAICS code.
    # I use duckdb for a bit of performance in these very large tables over pandas.
    frs_composting_fac_df = (
        duckdb.execute(
            (
                "SELECT frs_main.primary_name as name, frs_main.latitude83 as latitude, frs_main.longitude83 as longitude "  # noqa
                f"FROM (SELECT registry_id, primary_name, latitude83, longitude83 FROM '{frs_facility_path}') as frs_main "  # noqa
                f"JOIN (SELECT registry_id, naics_code FROM '{frs_naics_path}') AS frs_naics "  # noqa
                "ON frs_main.registry_id = frs_naics.registry_id "
                f"WHERE naics_code == {COMPOSTING_FRS_NAICS_CODE}"
            )
        )
        .df()
        .assign(formatted_fac_name=lambda df: name_formatter(df["name"]), source="frs")
    )
    frs_composting_fac_df

    biocycle_df = (
        pd.read_csv(biocycle_path)
        .rename(columns={"lat": "latitude", "lon": "longitude"})
        .assign(source="biocycle")
    )
    biocycle_df

    # google earth exports 3d points, covert them to 2d to avoid issues.
    def _drop_z(geom):
        return wkb.loads(wkb.dumps(geom, output_dimension=2))

    comp_council_gdf = (
        gpd.read_file(comp_council_path, driver="KML")
        .rename(columns=str.lower)
        .drop(columns="description")
        .assign(
            formatted_fac_name=lambda df: name_formatter(df["name"]),
            geometry=lambda df: df.geometry.apply(_drop_z),
            source="comp_council",
        )
    )
    comp_council_gdf

    # there is a two step process to get all facilities in one dataframe:
    # 1) put together the facilities that have lat/lon columns and make them
    # geodataframes,
    facility_concat_df = pd.concat(
        [frs_composting_fac_df, biocycle_df, excess_food_df]
    ).sort_values("formatted_fac_name")
    facility_concat_gdf = gpd.GeoDataFrame(
        facility_concat_df.drop(columns=["latitude", "longitude"]),
        geometry=gpd.points_from_xy(
            facility_concat_df["longitude"], facility_concat_df["latitude"]
        ),
        crs=4326,
    )
    # 2) concat these with the facilities data that are already geodataframes.
    facility_concat_2_gdf = pd.concat([comp_council_gdf, facility_concat_gdf])
    facility_concat_2_gdf = facility_concat_2_gdf[
        facility_concat_2_gdf.is_valid & ~facility_concat_2_gdf.is_empty
    ]
    facility_concat_2_gdf.normalize()

    # Remove duplicate facilities based on our spatial tolerance
    # if two or more facilities fall within the tolerance, only 1 location is kept, the
    # resulting centroid will be at the intersection of their buffers.
    final_facility_gdf = (
        facility_concat_2_gdf.to_crs("ESRI:102003")
        .assign(geometry=lambda df: df.buffer(DUPLICATION_TOLERANCE_M))
        .dissolve()
        .explode()
        .centroid.to_frame()
        .to_crs(4326)
        .sjoin(state_gdf[["geometry", "state_code"]])
        .drop(columns="index_right")
        .rename_geometry("geometry")
    )
    # %%
    # NOTE: DC has no facilities listed, but has 1 year (2012) with reported emissions.
    # Erin email note on DC: My understanding is that (industrial) composting in DC has
    # been outsources to MD. Since the inventory is trying to capture the distribution
    # of industrial composting emissions, let’s manually add a single facility location
    # for DC. Let’s use the Fort Totten Waste Transfer Station (lat/lon: 38.947624,
    # -77.001213). We can make a note of this in the data and assumptions document and
    # this assumption can be re-visited in later iterations.
    proxy_gdf = (
        pd.concat(
            [
                final_facility_gdf,
                state_gdf[state_gdf["state_code"] == "DC"][
                    ["state_code", "geometry"]
                ].assign(geometry=Point(-77.001213, 38.947624)),
            ]
        )
        .reset_index(drop=True)
        .assign(emis_kt=1)
    )
    proxy_gdf["rel_emi"] = proxy_gdf.groupby("state_code")["emis_kt"].transform(
        normalize
    )
    print("composting facilities with location: ", len(proxy_gdf))

    all_eq_df = (
        proxy_gdf.groupby("state_code")["rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(is_close=lambda df: (np.isclose(df["sum_check"], 1)))
    )
    all_eq_df

    if not all_eq_df["is_close"].all():
        raise ValueError("not all values are normed correctly!")
    # %%
    proxy_gdf.to_parquet(dst_path)


# %%
