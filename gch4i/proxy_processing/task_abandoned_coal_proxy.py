"""
Name:                  task_abandoned_coal_proxy.py
Date Last Modified:    2025-01-22
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               Process abandoned coal proxy data for emissions.
Input Files:           - Inventory Workbook: {ghgi_data_dir_path}/1B1a_abandoned_coal/
                        AbandonedCoalMines1990-2022_FRv1.xlsx
                       - MSHA: {sector_data_dir_path}/abandoned_mines/Mines.zip
                       - County: {global_data_dir_path}/tl_2020_us_county.zip
                       - State: {global_data_dir_path}/tl_2020_us_state.zip
Output Files:          - {proxy_data_dir_path}/abd_coal_proxy.parquet
"""

# %% Import Libraries

import datetime
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import pyarrow.parquet  # noqa
import osgeo  # noqa
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from pytask import Product, mark, task

from gch4i.config import (
    ghgi_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    max_year,
    min_year,
    proxy_data_dir_path
)
from gch4i.utils import name_formatter

pd.set_option("future.no_silent_downcasting", True)

# %% Pytask Function


@mark.persist
@task(id="abd_coal_proxy")
def task_get_abd_coal_proxy_data(
    inventory_workbook_path: Path = (
        ghgi_data_dir_path / "1B1a_abandoned_coal/AbandonedCoalMines1990-2022_FRv1.xlsx"
    ),
    msha_path: Path = sector_data_dir_path / "abandoned_mines/Mines.zip",
    county_path: str = global_data_dir_path / "tl_2020_us_county.zip",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "abd_coal_proxy.parquet"
    ),
):
    """
    Location information (e.g., latitude/longitude) for each mine is taken from the MSHA
    database.

    Mines in the AMM Opportunities database are matched to the MSHA databased
    based first on

    1. MSHA ID. In total, 430 mines are matched based on MSHA ID,


    If there is no match on the ID then the mine is attempted to be
    matched based on the mine...

    # NOTE: I have had better luck without the date for the intial match after ID
    2. name, state, county, and abandonment date.

    1 based on name, date, county, and state,

    6 on date, state, and county, and

    6 based on mine name, state, and county out of a total 544 mines.

    For the remaining 101 mines, these emissions are allocated across the county on
    record for that mine (based on the area in each grid cell relative to the total area
    in that county). This approach is also used for 6 mines where the reported Lat/Lon
    in the MSHA database does not match the state on record for that mine. In total,
    ~20% of total annual abandoned mine emissions are allocated to the county-level
    rather than specific mine location.
    """
    # %%
    # basin recode dictionary for both the ratios and mines list.
    basin_name_recode = {
        "Central Appl": 0,
        "Central Appl.": 0,
        "Illinois": 1,
        "Northern Appl": 2,
        "Northern Appl.": 2,
        "Warrior Basin": 3,
        "Warrior": 3,
        "Uinta": 4,
        "Raton": 4,
        "Arkoma": 4,
        "Piceance": 4,
        "Western Basins": 4,
    }

    # https://www.epa.gov/sites/default/files/2016-03/documents/amm_final_report.pdf
    # basin coefficients pulled from v2
    # order: CA, IL, NA, BW, WS
    basin_coef_dict = {
        "Flooded": [0.672, 0.672, 0.672, 0.672, 0.672],
        "Sealed": [0.000741, 0.000733, 0.000725, 0.000729, 0.000747],
        "Venting": [0.003735, 0.003659, 0.003564, 0.003601, 0.003803],
        "b_medium": [2.329011, 2.314585, 2.292595, 2.299685, 2.342465],
    }

    basin_coef_df = pd.DataFrame.from_dict(basin_coef_dict)

    # the set of functions to calculate emissions based on the status of the mine and
    # the basin it is in.
    def flooded_calc(df, basin, bc_df):
        # abdmines.loc[imine,'Active Emiss. (mmcfd) ']
        # * np.exp(-1*D_flooded[ibasin]*ab_numyrs)
        return df["active_emiss"] * np.exp(
            -1 * bc_df.loc[basin, "Flooded"] * df["years_closed"]
        )

    def venting_calc(df, basin, bc_df):
        # abdmines.loc[imine,'Active Emiss. (mmcfd) ']
        # * (1+b_medium[ibasin]*D_venting[ibasin]*ab_numyrs)
        # ** (-1/float(b_medium[ibasin]))
        return df["active_emiss"] * (
            1
            + bc_df.loc[basin, "b_medium"]
            * bc_df.loc[basin, "Venting"]
            * df["years_closed"]
        ) ** (-1 / bc_df.loc[basin, "b_medium"])

    def sealed_calc(df, basin, bc_df):
        # abdmines.loc[imine,'Active Emiss. (mmcfd) ']
        # * (1-0.8)
        # * (1+b_medium[ibasin]*D_sealed[ibasin]*ab_numyrs)
        # ** (-1/float(b_medium[ibasin]))
        return (
            df["active_emiss"]
            * (1 - 0.8)
            * (
                1
                + bc_df.loc[basin, "b_medium"]
                * bc_df.loc[basin, "Sealed"]
                * df["years_closed"]
            )
            ** (-1 / bc_df.loc[basin, "b_medium"])
        )

    def unknown_calc(df, basin, bc_df, r_df):
        em_flo = r_df.loc[basin, "Flooded"] * flooded_calc(df, basin, bc_df)
        em_ven = r_df.loc[basin, "Venting"] * venting_calc(df, basin, bc_df)
        em_sea = r_df.loc[basin, "Sealed"] * sealed_calc(df, basin, bc_df)
        return np.sum([em_flo, em_ven, em_sea], axis=0)

    # Create the ratios dataframe from the inventory workbook. Get the ratios of mines
    # that are sealed, venting, and flooded by basin and year. the rows to skip to find
    # each of the ratio tables.
    skip_row_list = [26, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30]
    ratio_list = []
    for year, skip_row in zip(range(min_year, max_year + 1), skip_row_list):

        ratios_df = (
            pd.read_excel(
                inventory_workbook_path,
                sheet_name=str(year),
                skiprows=skip_row,
                nrows=5,
            )
            .loc[:, ["Basin", "Sealed %", "Vented %", "Flooded %"]]
            .rename(
                columns={
                    "Sealed %": "Sealed",
                    "Vented %": "Venting",
                    "Flooded %": "Flooded",
                }
            )
            .assign(year=year)
        )
        ratio_list.append(ratios_df)
    ratios_df = (
        pd.concat(ratio_list).replace(basin_name_recode).set_index(["Basin", "year"])
    )
    ratios_normed_df = ratios_df.groupby("year").apply(
        lambda df: df.div(df.sum(axis=1), axis=0)
    )

    # %%
    # load the state, county, and mine data
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

    county_gdf = (
        gpd.read_file(county_path)
        .rename(columns=str.lower)
        .astype({"statefp": int})
        .merge(state_gdf[["state_code", "statefp"]], on="statefp")
        .assign(
            formatted_county=lambda df: name_formatter(df["name"]),
            formatted_state=lambda df: name_formatter(df["state_code"]),
        )
        .to_crs(4326)
    )

    # load the MSHA mine data
    with ZipFile(msha_path) as z:
        with z.open("Mines.txt") as f:
            msha_df = (
                pd.read_table(
                    f,
                    sep="|",
                    encoding="ISO-8859-1",
                    # usecols=["MINE_ID", "LATITUDE", "LONGITUDE"],
                )
                # EEM: this identifies whether the mine was a coal mine (C) or metal/non-metal mine
                .query("COAL_METAL_IND == 'C'")
                .dropna(subset=["LATITUDE", "LONGITUDE"])
                .assign(
                    formatted_name=lambda df: name_formatter(df["CURRENT_MINE_NAME"]),
                    formatted_county=lambda df: name_formatter(df["FIPS_CNTY_NM"]),
                    formatted_state=lambda df: name_formatter(df["STATE"]),
                    date_abd=lambda df: pd.to_datetime(df["CURRENT_STATUS_DT"]),
                )
                # .set_index("MINE_ID")
            )

    # make the mines data spatial
    msha_gdf = gpd.GeoDataFrame(
        msha_df.drop(columns=["LATITUDE", "LONGITUDE"]),
        geometry=gpd.points_from_xy(msha_df["LONGITUDE"], msha_df["LATITUDE"]),
        crs=4326,
    )
    # get only the mines that are in the lower 48 + DC
    # side effect that mines with invalid geometries are dropped
    msha_gdf = msha_gdf[msha_gdf.intersects(state_gdf.dissolve().geometry.iat[0])]

    ax = msha_gdf.plot(color="xkcd:scarlet", figsize=(10, 10), markersize=1)
    state_gdf.boundary.plot(ax=ax, color="xkcd:slate", lw=0.2, zorder=1)

    msha_cols = [
        "MINE_ID",
        "geometry",
        "CURRENT_MINE_STATUS",
        "CURRENT_CONTROLLER_BEGIN_DT",
    ]

    # get a crosswalk of records that we can't find via the mine DB or county, develop
    # a crosswalk for the 3 records we can't find a match for from these "county" names
    # to actual county names.
    # https://en.wikipedia.org/wiki/West,_West_Virginia
    # https://en.wikipedia.org/wiki/Dickenson_County,_Virginia
    # Finding the county for rosedale is a bit tricky. It's not a formal place?
    # https://www.mapquest.com/us/tennessee/rosedale-tn-282922764
    county_name_fixes = {
        "formatted_county": {
            "dickerson": "dickenson",
            "west": "wetzel",
            "rosedale": "anderson",
        }
    }

    # get the list of mines from the inventory workbook that are listed as abandoned.
    # these are the ones we will try to match to the MSHA location data.
    inventory_mine_df = (
        pd.read_excel(
            inventory_workbook_path,
            sheet_name="Mine List",
            skiprows=1,
            # nrows=115,
            usecols="B:M",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        .rename(
            columns={"active emiss. (mmcfd)": "active_emiss", "state": "state_code"}
        )
        .assign(
            MINE_ID=lambda df: pd.to_numeric(df["msha id"], errors="coerce")
            .fillna(0)
            .astype(int),
            formatted_name=lambda df: name_formatter(df["mine name"]),
            formatted_county=lambda df: name_formatter(df["county"]),
            formatted_state=lambda df: name_formatter(df["state_code"]),
            date_abd=lambda df: pd.to_datetime(df["date of aban."]),
        )
        .replace(county_name_fixes)
        .set_index("MINE_ID")
        .join(msha_gdf[msha_cols].set_index("MINE_ID"))
        .reset_index()
    )
    inventory_mine_df
    # %%
    # match inventory mines to the location data using a variety of strategies, in order
    # from the matching_attempts dictionary. The first match is by ID, then by the
    # dictionary.

    matching_attempts = {
        "name, date, county, state": [
            "formatted_name",
            "date_abd",
            "formatted_county",
            "formatted_state",
        ],
        "name, county, state": [
            "formatted_name",
            "formatted_county",
            "formatted_state",
        ],
        "date, county, state": ["date_abd", "formatted_county", "formatted_state"],
        "name, date": ["formatted_name", "date_abd"],
    }

    # First try to match by ID alone
    print(f"total mines: {inventory_mine_df.shape[0]}\n")
    matches = inventory_mine_df[~inventory_mine_df["geometry"].isna()]
    missing_geoms = inventory_mine_df[inventory_mine_df["geometry"].isna()]
    print(f"mines matched by ID: {matches.shape[0]}")
    print(f"mines still missing geom: {missing_geoms.shape[0]}\n")

    # now loop through the dictionary of matching attempts to try to match the remaining
    # mines.
    match_result_list = []
    match_result_list.append(matches)
    for match_name, col_list in matching_attempts.items():
        # match_attempt = missing_geoms.drop(columns=["geometry"]).merge(
        match_attempt = missing_geoms.drop(columns=msha_cols).merge(
            # msha_gdf.drop_duplicates(subset=col_list)[col_list + ["geometry"]],
            msha_gdf.drop_duplicates(subset=col_list)[col_list + msha_cols],
            # msha_gdf[col_list + ["geometry"]],
            left_on=col_list,
            right_on=col_list,
            how="left",
        )

        matches = match_attempt[~match_attempt["geometry"].isna()]
        missing_geoms = match_attempt[match_attempt["geometry"].isna()]
        match_result_list.append(matches)
        print(f"mines matched by {match_name}: {matches.shape[0]}")
        print(f"mines still missing geom: {missing_geoms.shape[0]}\n")

    match_attempt = missing_geoms.drop(columns="geometry").merge(
        county_gdf[["formatted_county", "formatted_state", "geometry"]],
        left_on=["formatted_county", "formatted_state"],
        right_on=["formatted_county", "formatted_state"],
        how="left",
    )
    matches = match_attempt[~match_attempt["geometry"].isna()]
    missing_geoms = match_attempt[match_attempt["geometry"].isna()]
    match_result_list.append(matches)
    print(f"mines matched by county and state to county polygon: {matches.shape[0]}")
    print(f"mines still missing geom: {missing_geoms.shape[0]}\n")

    matching_results = pd.concat(match_result_list)
    print(f"total geo match mines: {matching_results.shape[0]}\n")

    # %%

    # it appears the new workbooks has a clean version of 'Current Emissions Status'
    # called "simple status" so there is no need to recode that from the broader list
    # found in v2.
    all_mines_df = (
        matching_results.assign(
            # create column of recovering state
            recovering=lambda df: np.where(
                df["current model worksheet"].str.casefold().str.contains("recovering"),
                1,
                0,
            ),
            # reclass to basin number
            basin_nr=lambda df: df["coal basin"].replace(basin_name_recode),
            # get the reopen date
            reopen_date=lambda df: pd.to_datetime(df["CURRENT_CONTROLLER_BEGIN_DT"]),
            operating_status=lambda df: df["CURRENT_MINE_STATUS"],
            # filter down columns we need.
        )
        # calculate only for mines that have emissions
        .query("active_emiss > 0").loc[
            :,
            [
                "MINE_ID",
                "geometry",
                "state_code",
                "county",
                "basin_nr",
                "recovering",
                "reopen_date",
                "date_abd",
                "simple status",
                "active_emiss",
                "operating_status",
            ],
        ]
    )
    all_mines_df

    # carrying over the guidance from v2, we will remove mines that are listed as active
    #       NOTES:
    # 1)    Mines reopened in 2020 will not affect this notebook run for 2012-2018. #EEM: edit text
    # 2)    MSHA says mine 3600840 is active, but it is not in active mine GHGI
    #       workbook. It is in the abandoned mine workbook. abandoned in 1994.
    #       We keep it here.
    # 3)    Mine 4200079 is present in active mines notebook and also has abandoned
    #       emissions. However, its emissions are only ~0.8% of Utah's abandoned mines
    #       emissions. We keep it here.
    #       Old: We remove it in the flux calculation block for 2016-2018.
    # 4)    Mine 1100588 is listed as a refuse recovery mine in the active mines
    #       notebook, with coal production for the years 2012-2016. Refuse recovery mine
    #       emissions are not included in the active mining emissions estimates.
    #       This mine also has abandoned emissions and is in the abandoned GHGI workbook
    #       (close in 1995). We keep this here.
    #       #old: We remove it in the flux calculation block here for years 2014-2018.
    # 5)    Check to see if there are other mines listed below.

    # So we filter the active mines that closed in 2020 or later to handle in the yearly
    # emissions calculations.

    active_mines_df = all_mines_df.query(
        # "(operating_status == 'Active')"
        "(operating_status == 'Active') & (reopen_date.dt.year >= 2020)" #EEM - why is the re-open date set to 2020 and not a different year? Check that this isn't a hold-over from v2
    ).sort_values("reopen_date")
    print(
        "mines that are in the abandoned workbook but "
        f"listed as active in the mine db: {len(active_mines_df)}"
    )
    active_mines_df

    # questionable mines are those that have a reopen date after the date of
    # abandonment, regardless of their status. There are 52 mines total that fall into
    # this grey zone, include the 6 that are listed as active.
    # questionable_mines = all_mines_df.query(
    #     "reopen_date > date_abd"
    # ).sort_values("MINE_ID")

    # remove the active mines from the abandoned mines
    abandoned_mines_df = all_mines_df.drop(index=active_mines_df.index)

    # %%
    # previously year days were recalculated for every year to calculate the fraction
    # of years a mine way closed. I think a better approach would be to assign a
    # constant that roughly equals the number of days in a year.
    year_days = 365.25

    # Ensure active_mines_df has a datetime object for reopen_date
    active_mines_df["reopen_date"] = pd.to_datetime(active_mines_df["reopen_date"], errors="coerce")

    result_list = []
    # for each year we have mine data, calculate emissions based on the basin, mine
    # status, and the number of years closed.
    reopened_mine_list = []
    for year in range(min_year, max_year + 1):

        # get the normalized ratios of mine status by year
        yearly_ratios_normed_df = ratios_normed_df.loc[year].droplevel(-1)

        # check that the ratios sum to 1 for each basin
        if (
            not yearly_ratios_normed_df.sum(axis=1)
            .apply(lambda x: np.isclose(x, 1))
            .all()
        ):
            raise ValueError("Ratios do not sum to 1")

        # # get the number of days in the year
        # month_days = [calendar.monthrange(year, x)[1] for x in range(1, 13)]
        # year_days = np.sum(month_days)

        # this year date to calc relative emissions
        # NOTE: this is different from the v2 notebook where the date was 07/02
        # We can calculate the actual fraction of emissions for a given year.
        calc_date = datetime.datetime(year=year, month=12, day=31)

      #EEM: is the code below legacy from v2? If so, please delete
        # calculate the number of days closed relative to 07/02 of this year?
        # XXX: why not calculate the entire year?
        # if the mine closed this year, give it special treatment where the
        # number of days closed is 1/2 of the number of days closed relative to
        # our date of 07/02 the logic here is: if the mine closed in this year,
        # the number of days closed is equal to 1/2 the days closed relative to
        # 07/02.
        # calc_date = datetime.datetime(year=year, month=7, day=2)

        # get the mines that are abandoned this year or earlier
        year_aban_df = abandoned_mines_df.query(
            "(date_abd.dt.year <= @calc_date.year)"
            # "& (reopen_date.dt.year <= @calc_date.year)"
        )
        year_aban_df  = year_aban_df [
            ~year_aban_df .MINE_ID.isin(reopened_mine_list)
        ]

        # FOR REFERENCE: mines that were abandoned this year
        aban_this_year_df = year_aban_df.query("date_abd.dt.year == @calc_date.year")

        # FOR REFERENCE: abandoned mines that were reopened this year
        aban_reopen_this_year_df = year_aban_df.query(
            "(reopen_date.dt.year == @calc_date.year)"
        )

        reopened_mine_list.extend(
            aban_reopen_this_year_df.MINE_ID.to_list()
        )
        # these are mines that are listed as active, but were closed this year
        # so we take these mines and calculate the days closed as the difference
        # of days from when it when it was abandoned to the day it reopened this year.
        # if the mine was opened this year, we subtract out the number of days it was
        # operational from the days closed
        active_but_closed_this_year_df = active_mines_df.query(
            "(reopen_date >= @calc_date)"
        )

        print(f"total mines abandoned {year}: {len(year_aban_df)}")
        print(f"mines abandoned in this year {year}: {len(aban_this_year_df)}")
        print(
            f"mines abandoned reproting reopen this year {year}: {len(aban_reopen_this_year_df)}"
        )
        # print(aban_reopen_this_year_df.MINE_ID.to_list())
        # print()

        print(
            f"num actives mines that closed this year: {len(active_but_closed_this_year_df)}"
        )

        # combine the abandoned and newly reopened mines for this year
        year_mines_df = pd.concat(
            [year_aban_df, active_but_closed_this_year_df]
        ).assign(
            # assign the current year for calculations
            year=year,
            days_closed=lambda df: (calc_date - df["date_abd"]),
            # days_closed=lambda df: np.where(
            #     df["date_abd"].dt.year == year,
            #     -((calc_date - df["date_abd"]) / 2),
            #     calc_date - df["date_abd"],
            # ),
            # calculate the number of years closed
            years_closed=lambda df: (df["days_closed"].dt.days / year_days),
            # create an empty column to hold the results
            mine_emi=0,
        )

        # these are mines that are listed as active, but were reopened this year
        # so we take these mines and calculate the days closed as the difference
        # of days from when it when it was abandoned to the day it reopened this year.
        # if the mine was opened this year, we subtract out the number of days it was
        # operational from the days closed
        year_active_mines_df = active_mines_df.query(
            "(date_abd < @calc_date) & (reopen_date.dt.year >= @calc_date.year)"
        ).assign(
            year=year,
            operating_days=lambda df: np.where(
                df["reopen_date"].dt.year.eq(year),
                (calc_date - df["reopen_date"]).dt.days,
                0
            ),
            days_closed=lambda df: (calc_date - df["date_abd"]).dt.days - df["operating_days"],
            years_closed=lambda df: df["days_closed"] / year_days,
            # create an empty column to hold the results
            mine_emi=0,
        )

        # combine the abandoned and newly reopened mines for this year
      # EEM: where is year_abandoned_mines_df defined? This doesn't seem to appear anywhere else in the file
        year_mines_df = pd.concat([year_abandoned_mines_df, year_active_mines_df])

        # we now calculate the emissions for each mine based on the status of the mine
        # and the basin it is in.

        data_list = []
        for (basin, status), data in year_mines_df.groupby(
            ["basin_nr", "simple status"]
        ):
            if status == "Flooded":
                data["mine_emi"] = flooded_calc(data, basin, basin_coef_df)
            if status == "Venting":
                data["mine_emi"] = venting_calc(data, basin, basin_coef_df)
            if status == "Sealed":
                data["mine_emi"] = sealed_calc(data, basin, basin_coef_df)
            # if the status of the mine is unknown, we calculate the emissions based on
            # the fraction of mines that are sealed, venting, and flooded in the basin.
            if status == "Unknown":
                data["mine_emi"] = unknown_calc(
                    data, basin, basin_coef_df, yearly_ratios_normed_df
                )
            data_list.append(data)
        res_df = pd.concat(data_list)
        # mines that are recovering are assigned 0 emissions
        res_df.loc[res_df["recovering"].eq(1), "mine_emi"] = 0
        result_list.append(res_df)

    result_df = pd.concat(result_list, ignore_index=True)

    # QC all recovering mines have 0 emissions
    if (
        not result_df[result_df["recovering"].eq(1)]
        .groupby("year")["mine_emi"]
        .sum()
        .eq(0)
        .all()
    ):
        raise ValueError("Recovering mines should have 0 emissions")

    # result_df.head()

    # %%
    # some visuals to check the data
    sns.relplot(data=result_df, y="mine_emi", x="year", hue="state_code", kind="line")
    # sns.relplot(data=result_df, y="mine_emi", x="year", kind="line")

    # some QC checks I'll leave in here for reference
    # the emission by state and year
    # result_df.groupby(["year", "state_code"])["mine_emi"].sum()

    # check the emissions based on the basin and status
    # result_df.groupby(["basin_nr", "simple status"])[
    #     "mine_emi"
    # ].sum().reset_index()

    # big list of checks for state, year, basin, status
    # result_df.groupby(["state_code", "year", "basin_nr", "simple status"])[
    #     "mine_emi"
    # ].sum().reset_index()
    # %%

    # format the final proxy data by getting only the columns we need.
    proxy_gdf = gpd.GeoDataFrame(result_df, crs=4326).loc[
        :, ["MINE_ID", "state_code", "year", "mine_emi", "geometry"]
    ]

    # Remove mines that have 0 emissions
    proxy_gdf = proxy_gdf[proxy_gdf["mine_emi"] > 0]

    # calculate the relative emissions by state and year for allocation.
    proxy_gdf["rel_emi"] = proxy_gdf.groupby(["state_code", "year"])[
        "mine_emi"
    ].transform(lambda x: x / x.sum())

    # %%
    # check that relative values by state and year sum to 1.
    if (
        not proxy_gdf.groupby(["state_code", "year"])["rel_emi"]
        .sum()
        .apply(lambda x: np.isclose(x, 1))
        .all()
    ):
        raise ValueError("relative emissions do not sum to 1")
    # %%
    # visual check of the final proxy data
    ax = state_gdf.boundary.plot(color="xkcd:slate", lw=0.2, figsize=(10, 10))
    proxy_gdf.plot(column="rel_emi", markersize=5, legend=False, ax=ax)
    # %%
    # save the final proxy data
    proxy_gdf.to_parquet(output_path)

# %%
