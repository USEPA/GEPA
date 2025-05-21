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


# %% Assign Constant Variables
COMPOSTING_FRS_NAICS_CODE = 562219
DUPLICATION_TOLERANCE_M = 2_500  # 2.5 km

composting_dir = sector_data_dir_path / "composting"
# %%


# %% Pytask Function
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
    """
    Generate proxy data for composting facilities by combining multiple data sources and
    removing duplicate facilities based on spatial proximity.

    Args:
        excess_food_op_path (Path): Path to the excess food composting facilities data.
        frs_facility_path (Path): Path to the FRS facility data.
        frs_naics_path (Path): Path to the FRS NAICS data.
        biocycle_path (Path): Path to the biocycle composting facilities data.
        comp_council_path (Path): Path to the compost council composting facilities data.
        state_geo_path (Path): Path to the state geography data.
        dst_path (Path): Path to save the output data.

    Returns:
        Creates a parquet file containing facility locations with state codes.
    """
    # read in state geometries
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

    # This geom will be used to filter facilities spatially
    STATE_UNION = state_gdf.to_crs("ESRI:102003").union_all()
    # %%

    # read in facilities data from all the sources

    def make_gdf(in_df):
        out_gdf = gpd.GeoDataFrame(
            in_df.drop(columns=["latitude", "longitude"]),
            geometry=gpd.points_from_xy(in_df["longitude"], in_df["latitude"]),
            crs=4326,
        ).to_crs("ESRI:102003")
        out_gdf = out_gdf[out_gdf.is_valid & ~out_gdf.is_empty]
        out_gdf = out_gdf[out_gdf.intersects(STATE_UNION)].loc[
            :, ["source", "geometry"]
        ]
        return out_gdf

    def read_comp_coucil_data(in_path):
        # google earth exports 3d points, covert them to 2d to avoid issues.
        def _drop_z(geom):
            return wkb.loads(wkb.dumps(geom, output_dimension=2))

        out_gdf = (
            gpd.read_file(in_path, driver="KML")
            .rename(columns=str.lower)
            .drop(columns="description")
            .assign(
                formatted_fac_name=lambda df: name_formatter(df["name"]),
                geometry=lambda df: df.geometry.apply(_drop_z),
                source="comp_council",
            )
            .to_crs("ESRI:102003")
        )

        out_gdf = out_gdf[out_gdf.is_valid & ~out_gdf.is_empty]
        out_gdf = out_gdf[out_gdf.intersects(STATE_UNION)].loc[
            :, ["source", "geometry"]
        ]

        return out_gdf

    def read_biocycle_data(in_path):
        out_df = (
            pd.read_csv(in_path)
            .rename(columns={"lat": "latitude", "lon": "longitude"})
            .assign(source="biocycle")
        )
        out_gdf = make_gdf(out_df)
        return out_gdf

    def read_frs_data(frs_path, naics_path):
        out_df = (
            duckdb.execute(
                (
                    "SELECT frs_main.primary_name as name, frs_main.latitude83 as latitude, frs_main.longitude83 as longitude "  # noqa
                    f"FROM (SELECT registry_id, primary_name, latitude83, longitude83 FROM '{frs_path}') as frs_main "  # noqa
                    f"JOIN (SELECT registry_id, naics_code FROM '{naics_path}') AS frs_naics "  # noqa
                    "ON frs_main.registry_id = frs_naics.registry_id "
                    f"WHERE naics_code == {COMPOSTING_FRS_NAICS_CODE}"
                )
            )
            .df()
            .assign(
                formatted_fac_name=lambda df: name_formatter(df["name"]), source="frs"
            )
        )
        out_gdf = make_gdf(out_df)

        return out_gdf

    def read_excess_food_data(in_path):
        out_df = (
            pd.read_excel(
                in_path,
                sheet_name="Data",
                usecols=["Name", "Latitude", "Longitude"],
            )
            .rename(columns=str.lower)
            .assign(
                formatted_fac_name=lambda df: name_formatter(df["name"]),
                source="ex_food",
            )
        )
        out_gdf = make_gdf(out_df).loc[:]

        return out_gdf

    epa_gdf = read_excess_food_data(excess_food_op_path)
    biocycle_gdf = read_biocycle_data(biocycle_path)
    frs_gdf = read_frs_data(frs_facility_path, frs_naics_path)
    comp_council_gdf = read_comp_coucil_data(comp_council_path)
    # %%
    # plot out the data to see what we have
    ax = state_gdf.to_crs("ESRI:102003").boundary.plot(color="black", linewidth=0.5)
    epa_gdf.plot(ax=ax, color="blue", markersize=5, label="EPA Facilities")
    biocycle_gdf.plot(ax=ax, color="red", markersize=5, label="Biocycle Facilities")
    comp_council_gdf.plot(
        ax=ax, color="green", markersize=5, label="Composting Council Facilities"
    )
    frs_gdf.plot(ax=ax, color="orange", markersize=5, label="FRS Facilities")
    ax.set_title("Composting Facilities")

    # %%

    # for both the biocycle and composting council datasets, we have less confidence in
    # the location data, therefore, we want to compare the locations of facilities with
    # the EPA data and remove any facilities that are within 2.5 km of an EPA facility.

    # We give first preference to the EPA data, so we first define the final faclity
    # list with it.
    final_fac_gdf = epa_gdf.copy()
    print(f"starting fac count:            {len(final_fac_gdf)}")
    print("=" * 40)

    # remove any facilities that are within 2.5 km of an EPA facility and the cascading
    # join of adding new facilities to the final dataset. The order here gives
    # preference to the EPA data, then biocycle, then composting council, and finally
    # FRS.
    fac_buffer = final_fac_gdf.buffer(DUPLICATION_TOLERANCE_M).union_all()
    for other_gdf in [biocycle_gdf, comp_council_gdf, frs_gdf]:
        source = other_gdf["source"].iloc[0]
        print(f"processing source:             {source}")
        print("-" * 40)
        print(f"facility count:                {len(other_gdf)}")
        # check if the other gdf intersects with the final facilities data.
        intersect_mask = other_gdf.intersects(fac_buffer)
        # get the number of facilities that DO intersect with the EPA buffer
        # and report their count, these will not be used.
        drop_gdf = other_gdf[intersect_mask]
        # get the facilities that are outside the buffer and should be added to the
        # final_fac_gdf
        add_gdf = other_gdf[~intersect_mask]

        print(f"duplicate facility count:      {len(drop_gdf)}")
        print(f"adding new facility count:     {len(add_gdf)}")
        # add the intersecting facilities to the final_fac_gdf
        final_fac_gdf = pd.concat([final_fac_gdf, add_gdf])
        print("-" * 40)
        print(f"new total proxy facilities:    {len(final_fac_gdf)}")
        print()
        # create a new buffer for the final_fac_gdf
        fac_buffer = final_fac_gdf.buffer(DUPLICATION_TOLERANCE_M).union_all()
    print("=" * 40)

    # NOTE: DC has no facilities listed, but has 1 year (2012) with reported emissions.
    # Erin email note on DC: My understanding is that (industrial) composting in DC has
    # been outsources to MD. Since the inventory is trying to capture the distribution
    # of industrial composting emissions, let’s manually add a single facility location
    # for DC. Let’s use the Fort Totten Waste Transfer Station (lat/lon: 38.947624,
    # -77.001213). We can make a note of this in the data and assumptions document and
    # this assumption can be re-visited in later iterations.
    dc_gdf = gpd.GeoDataFrame(
        {"source": "manual", "geometry": [Point(-77.001213, 38.947624)]}, crs=4326
    )

    print("adding DC facility")
    print("=" * 40)
    # add the DC facility and join the state_code onto the facilities
    proxy_gdf = (
        pd.concat([final_fac_gdf.to_crs(4326), dc_gdf])
        .reset_index(drop=True)
        .assign(emis_kt=1)
        .sjoin(state_gdf[["state_code", "geometry"]])
        .drop(columns="index_right")
    )
    print(f"final fac count:               {len(proxy_gdf)}")
    # %%
    # normalize the emiissions by equal allocation to all facilites within a state
    # Since there is no year to these data, there is no yearly normalization applied.
    proxy_gdf["rel_emi"] = proxy_gdf.groupby("state_code")["emis_kt"].transform(
        normalize
    )

    # check that the emissions are normalized correctly
    # this should be 1 for all states
    all_eq_df = (
        proxy_gdf.groupby("state_code")["rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(is_close=lambda df: (np.isclose(df["sum_check"], 1)))
    )

    if not all_eq_df["is_close"].all():
        raise ValueError("not all values are normed correctly!")
    # %%
    # final plot of the data by source
    ax = state_gdf.boundary.plot(color="xkcd:slate", linewidth=0.5)
    proxy_gdf.plot(
        "source", cmap="Set3", markersize=1, categorical=True, legend=True, ax=ax
    )
    ax.set_title("Composting proxy Facilities")
    # %%
    proxy_gdf.to_parquet(dst_path)


# %%
