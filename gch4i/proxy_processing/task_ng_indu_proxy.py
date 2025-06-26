"""
Name:                   task_stat_comb_proxy.py
Date Last Modified:     2025-06-10
Authors Name:           J. Bollenbacher (RTI International)
Purpose:                Mapping industrial ng emissions within the continental US
Input Files:            - GHGRP Subpart C: GEPA_Stat_Path / "InputData/GHGRP/GHGRP_SubpartCEmissions_2010-2023.csv"
                        - GHGRP Subpart D: GEPA_Stat_Path / "InputData/GHGRP/GHGRP_SubpartDEmissions_2010-2023.csv"
                        - GHGRP Subpart D Locations: GEPA_Stat_Path / "InputData/GHGRP/GHGRP_FacilityInfo_2010-2023.csv"

Output Files:           - {proxy_data_dir_path} / ng_indu_proxy.parquet
Notes:                  -
"""

from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
from pytask import Product, mark, task

from gch4i.config import global_data_dir_path, proxy_data_dir_path
from gch4i.proxy_processing.task_stat_comb_proxy import (
    GHGRP_subC_inputfile,
    GHGRP_subD_inputfile,
    GHGRP_subDfacility_loc_inputfile,
    create_raw_indu_proxy,
)


@mark.persist
@task(id="ng_indu_proxy")
def task_get_reporting_ng_indu_proxy_data(
    subpart_C=GHGRP_subC_inputfile,
    subpart_D=GHGRP_subD_inputfile,
    facility_path=GHGRP_subDfacility_loc_inputfile,
    state_path=global_data_dir_path / "tl_2020_us_state.zip",
    reporting_ng_indu_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "ng_indu_proxy.parquet",
):

    proxy_gdf = create_raw_indu_proxy(
        subpart_C, subpart_D, facility_path, reporting_ng_indu_proxy_output_path
    )

    # filter proxy_gdf so it only contains points which fall within the continental US
    # geometry
    continental_us_gdf = (  # get continential US geometries from state geometries
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
        .dissolve()  # combine all states into single geometry
        .reset_index(drop=True)[
            ["geometry"]
        ]  # select only the geometry column for spatial filtering
    )
    proxy_gdf = gpd.sjoin(
        proxy_gdf, continental_us_gdf, how="inner", predicate="within"
    ).drop(columns=["index_right"])

    # Normalize relative emissions to sum to 1 for each year and state
    # Drop state-years with 0 total volume
    proxy_gdf = proxy_gdf.groupby(["state_code", "year"]).filter(
        lambda x: x["ch4_flux"].sum() > 0
    )
    # Normalize annual state emissions to sum to 1
    proxy_gdf["annual_rel_emi"] = proxy_gdf.groupby(["year", "state_code"])[
        "ch4_flux"
    ].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    # Drop the original ch4_flux column
    proxy_gdf = proxy_gdf.drop(columns=["ch4_flux"])

    proxy_gdf.to_parquet(reporting_ng_indu_proxy_output_path)
    return None
