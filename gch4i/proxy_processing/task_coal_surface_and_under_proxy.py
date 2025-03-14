"""
Name:                   task_coal_surface_and_under_proxy.py
Date Last Modified:     2025-01-27
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This script produces proxies for underground and surface coal
                        mines, and post mining for both. The proxies are based on the
                        production data from the EIA and the MSHA mines database. A
                        filter is applied to the mines database to extract either the
                        underground or surface mines. The production data is then joined
                        with the mines database to get the production data for each
                        mine.
Input Files:            - {ghgi_data_dir_path} / "1B1a_coal_mining_underground/
                            Coal_90-22_FRv1-InvDBcorrection.xlsx"
                        - {ghgi_data_dir_path} / "1B1a_coal_mining_surface/
                            Coal_90-22_FRv1-InvDBcorrection.xlsx"
                        - {sector_data_dir_path} / "abandoned_mines/Mines.zip"
                        - {global_data_dir_path} / "tl_2020_us_state.zip"
                        - {EIA_dir_path} / "coalpublic{year}.xls"
Output Files:           - {proxy_data_dir_path} /
                            "coal_{mine_type.lower()}_proxy.parquet"
                        - {proxy_data_dir_path} /
                            "coal_post_{mine_type.lower()}_proxy.parquet"

NOTE: The 2021 and recent EIA files had to be converted manually to the .xlsx format.
These files are actually XML Spreadsheet 2003 format with an extension of .xls. Pandas
could not read the  format. It was not worth the effort to write a parser for this
format or deal with the read_xml method.
# https://stackoverflow.com/questions/33470130/read-excel-xml-xls-file-with-pandas
# https://pandas.pydata.org/docs/reference/api/pandas.read_xml.html
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import pyarrow.parquet  # noqa
import osgeo  # noqa
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pytask import Product, mark, task
from tqdm.auto import tqdm

from gch4i.config import (
    ghgi_data_dir_path,
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
from gch4i.utils import download_url, normalize


# %% Set Constants & Paths
pd.set_option("future.no_silent_downcasting", True)
pd.set_option("float_format", "{:f}".format)

eia_col_dict = {
    "Year": "year",
    "MSHA ID": "MINE_ID",
    "Production (short tons)": "production",
    "Mine State": "mine_state",
    "Mine County": "county_name",
    "Mine Type": "mine_type",
    "Mine Status": "mine_status",
}
status_filter = ["Active"]
# status_filter = ["Active", "Active, men working, not producing"]

# I get these manually from the list of mines by year in the inventory workbook
# These are used to filter the mines list to only the mines that are in the inventory
# workbook since there are other tables in those sheets below the mines list.
inv_ug_mine_count_by_year = {
    2012: 117,
    2013: 205,
    2014: 178,
    2015: 220,
    2016: 163,
    2017: 162,
    2018: 164,
    2019: 167,
    2020: 231,
    2021: 203,
    2022: 209,
}


# Million cubic ft (mmcf) to Tg conversion factor - Source: EPA spreadsheet,
# 'CM Emissions Summary' cell C 40.
mmcf_to_Gg = 51921

EIA_dir_path = sector_data_dir_path / "coal/EIA"

mine_list_filter_cols = [
    "MSHA Mine ID",
    "District No.",
    "Mine Name",
    "State",
    # "County",
    # "Company Name",
    # "Basin",
]


for year in years:
    eia_name = f"coalpublic{year}.xls"
    eia_file_path = EIA_dir_path / eia_name
    url = f"https://www.eia.gov/coal/data/public/xls/coalpublic{year}.xls"

    @mark.persist
    @task(
        id=f"download_eia_data_{year}", kwargs={"url": url, "out_path": eia_file_path}
    )
    def task_download_eia_data(url: str, out_path: Annotated[Path, Product]) -> None:
        download_url(url, out_path)


# NOTE: the 2021 and recent EIA files had to be converted manually to the .xlsx format
# because pandas could not read the .xls format. These files are actually XML
# Spreadsheet 2003 format. It was not worth the effort to write a parser for this format
# or deal with the read_xml method.
# https://stackoverflow.com/questions/33470130/read-excel-xml-xls-file-with-pandas
# https://pandas.pydata.org/docs/reference/api/pandas.read_xml.html
eia_input_paths = []
for year in years:
    if year >= 2021:
        eia_name = f"coalpublic{year}.xlsx"
    else:
        eia_name = f"coalpublic{year}.xls"
    eia_file_path = EIA_dir_path / eia_name
    eia_input_paths.append(eia_file_path)

# %%

CNTY_GEO_PATH: Path = global_data_dir_path / "tl_2020_us_county.zip"


mine_types = ["Underground", "Surface"]
param_dict = {}
for mine_type in mine_types:

    if mine_type == "Underground":
        ghgi_file_name = (
            "1B1a_coal_mining_underground/Coal_90-22_FRv1-InvDBcorrection.xlsx"
        )
    if mine_type == "Surface":
        ghgi_file_name = (
            "1B1a_coal_mining_surface/" "Coal_90-22_FRv1-InvDBcorrection.xlsx"
        )

    param_dict[mine_type] = dict(
        mine_type=mine_type,
        inventory_workbook_path=(ghgi_data_dir_path / ghgi_file_name),
        msha_path=sector_data_dir_path / "abandoned_mines/Mines.zip",
        eia_paths=eia_input_paths,
        state_path=global_data_dir_path / "tl_2020_us_state.zip",
        output_path_under=(
            proxy_data_dir_path / f"coal_{mine_type.lower()}_proxy.parquet"
        ),
        output_path_under_post=(
            proxy_data_dir_path / f"coal_post_{mine_type.lower()}_proxy.parquet"
        ),
    )


(
    mine_type,
    inventory_workbook_path,
    msha_path,
    eia_paths,
    state_path,
    output_path_coal,
    output_path_coal_post,
) = param_dict["Underground"].values()


# %% Pytask Function
for _id, kwargs in param_dict.items():

    @mark.persist
    @task(id=f"coal_{_id}_proxy", kwargs=kwargs)
    def task_coal_proxy(
        mine_type: str,
        inventory_workbook_path: Path,
        msha_path: Path,
        eia_paths: list[Path],
        state_path: Path,
        output_path_coal: Annotated[Path, Product],
        output_path_coal_post: Annotated[Path, Product],
    ) -> None:
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
            gpd.read_file(CNTY_GEO_PATH)
            .loc[:, ["NAME", "STATEFP", "COUNTYFP", "geometry", "GEOID"]]
            .rename(columns=str.lower)
            .rename(columns={"name": "county_name"})
            .astype({"statefp": int, "countyfp": int})
            .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
            .assign(
                fips=lambda df: df["geoid"].astype(str).str.zfill(5),
            )
            .to_crs(4326)
            .merge(
                state_gdf[["state_name", "statefp", "state_code"]],
                on="statefp",
                how="left",
            )
            # .assign(rep_point=lambda df: df.representative_point())
            # .set_geometry("rep_point")
            # .sjoin(state_gdf, how="left")
            # .set_geometry("geometry")
            # .drop(columns=["rep_point"])
            .assign(
                county_name=lambda df: df["county_name"].str.casefold(),
                state_name=lambda df: df["state_name"].str.casefold(),
            )
        )

        # %%
        # load the MSHA mine data
        with ZipFile(msha_path) as z:
            with z.open("Mines.txt") as f:
                msha_df = (
                    pd.read_table(
                        f,
                        sep="|",
                        encoding="ISO-8859-1",
                        usecols=[
                            "MINE_ID",
                            "LATITUDE",
                            "LONGITUDE",
                            "CURRENT_MINE_TYPE",
                            "CURRENT_MINE_STATUS",
                            "FIPS_CNTY_CD",
                            "BOM_STATE_CD",
                        ],
                    )
                    .astype({"MINE_ID": int})
                    .set_index("MINE_ID")
                    .query("CURRENT_MINE_TYPE == @mine_type")
                    .assign(
                        fips=lambda df: df["BOM_STATE_CD"].astype(str).str.zfill(2)
                        + df["FIPS_CNTY_CD"].astype(str).str.zfill(3)
                    )
                )

        # this is used to get location data onto inventory mines and EIA mines via
        # MINE ID
        msha_gdf = (
            gpd.GeoDataFrame(
                msha_df,
                geometry=gpd.points_from_xy(msha_df.LONGITUDE, msha_df.LATITUDE),
                crs=4326,
            )
            .sjoin(
                county_gdf.set_index("state_code")[
                    ["state_name", "county_name", "fips", "geometry"]
                ],
                how="inner",
            )
            .assign(state_name=lambda df: df.state_name.str.strip().str.casefold())
        )

        msha_gdf = msha_gdf[(msha_gdf.is_valid) & (~msha_gdf.is_empty)].copy()
        msha_no_geo_df = msha_df[~msha_df.index.isin(msha_gdf.index)]
        msha_w_cnty_gdf = (
            county_gdf[["fips", "geometry"]]
            .merge(msha_no_geo_df.reset_index(), on="fips")
            .set_index("MINE_ID")
        )
        msha_w_cnty_gdf = msha_w_cnty_gdf[
            (msha_w_cnty_gdf.is_valid) & (~msha_w_cnty_gdf.is_empty)
        ].copy()
        msha_no_geo_df = msha_no_geo_df[
            ~msha_no_geo_df.index.isin(msha_w_cnty_gdf.index)
        ]
        msha_w_geo_gdf = pd.concat([msha_gdf, msha_w_cnty_gdf])

        _, ax = plt.subplots(dpi=300, figsize=(10, 10))
        msha_gdf.plot(color="xkcd:lavender", markersize=1, ax=ax)
        msha_w_cnty_gdf.plot(
            color="xkcd:orange", markersize=1, ax=ax, zorder=-1, alpha=0.5
        )
        state_gdf.boundary.plot(ax=ax, lw=1, color="xkcd:slate")
        plt.show()

        # Check the sums of all the geodataframes equals the original msha_df
        total_msha_count = msha_df.shape[0]
        msha_w_geo_count = msha_w_geo_gdf.shape[0]
        msha_no_geo_count = msha_no_geo_df.shape[0]

        print(f"MSHA mines w/ point geo:    {msha_gdf.index.nunique():,}")
        print(f"MSHA mines w/ county:       {msha_w_cnty_gdf.index.nunique():,}")
        print("-" * 50)
        print(f"MSHA mines w/ comb geo:     {msha_w_geo_gdf.index.nunique():,}")
        print()
        print(f"MSHA mines w/ comb geo:     {msha_w_geo_gdf.index.nunique():,}")
        print(f"MSHA mines w/o geo:         {msha_no_geo_df.index.nunique():,}")
        print("-" * 50)
        print(f"total MSHA mines:           {msha_df.shape[0]:,}")

        if not total_msha_count == (msha_w_geo_count + msha_no_geo_count):
            print(
                f"Sum of geodataframes does not match original MSHA dataframe: "
                f"{msha_w_geo_count} != {msha_no_geo_count} + {msha_w_geo_count}"
            )
        else:
            print("Sum of all geodataframes equals the original MSHA dataframe.")

        # %% GET EIA DATA

        # this EIA based proxy primarily relies on EIA data and normalizes around a
        # basin weighted production. EIA does not account for all the states in the
        # inventory that have emissions, so we fill in missing states with MSHA data.

        # For surface, this EIA-based proxy serves both the mining and post mining
        # for underground, we use the inventory mines lists for mining and the EIA data
        # for post
        eia_mines_list = []
        for year in years:
            eia_file_path = [x for x in eia_paths if str(year) in x.name][0]

            # rename some columns for easier use, get only the correct mine type
            e_df = (
                pd.read_excel(
                    eia_file_path, skiprows=3, usecols=eia_col_dict.keys()
                ).rename(columns=eia_col_dict)
                # .query("(mine_type == @mine_type)")
                .query("(mine_type == @mine_type) & (production > 0)")
                # .query("(mine_type == @mine_type) & (mine_status.isin(@status_filter))")
            )
            # print(f"are there duplicated mines: {e_df.MINE_ID.duplicated().any()}")
            eia_mines_list.append(e_df)
        eia_mines_df = (
            pd.concat(eia_mines_list)
            .rename(columns=eia_col_dict)
            .set_index("MINE_ID")
            .assign(
                source="eia",
                state_name=lambda df: df.mine_state.str.split("(")
                .str[0]
                .str.strip()
                .str.casefold(),
                county_name=lambda df: df.county_name.str.casefold(),
            )
            .query("state_name != 'alaska'")
        )
        eia_mines_df

        # %%

        unique_eia_mines = (
            eia_mines_df.reset_index()[["MINE_ID", "state_name", "county_name"]]
            .drop_duplicates()
            .sort_values("MINE_ID")
            .set_index("MINE_ID")
        )
        unique_eia_mines
        # %%
        duplicate_mine_ids = unique_eia_mines[
            unique_eia_mines.index.duplicated(keep=False)
        ]
        duplicate_mine_ids

        # %%

        print(f"total EIA mines: {eia_mines_df.shape[0]:,}")
        count_not_in_msha = eia_mines_df[
            ~eia_mines_df.index.isin(msha_gdf.index)
        ].shape[0]
        print(f"EIA mines not in MSHA: {count_not_in_msha:,}")

        eia_mines_msha_geo_gdf = msha_gdf[
            ["state_name", "state_code", "geometry"]
        ].join(
            eia_mines_df[
                [
                    "mine_state",
                    "state_name",
                    "county_name",
                    "mine_status",
                    "production",
                    "year",
                ]
            ],
            how="right",
            rsuffix="_eia",
            # lsuffix="_msha",
        )
        print(f"EIA mines w/ MSHA points: {eia_mines_msha_geo_gdf.shape[0]:,}")

        invalid_points_mask = (~eia_mines_msha_geo_gdf.is_valid) & (
            eia_mines_msha_geo_gdf.is_empty
        )
        mismatched_mine_mask = (
            eia_mines_msha_geo_gdf["state_name_eia"]
            != eia_mines_msha_geo_gdf["state_name"]
        )

        # Get mines where eia_state_name does not equal msha_state_name
        mismatched_state_mines = eia_mines_msha_geo_gdf[mismatched_mine_mask]
        # removed the mismatched state mines from the eia_mines_msha_geo_gdf
        eia_mines_msha_geo_gdf = eia_mines_msha_geo_gdf[
            (~invalid_points_mask) & (~mismatched_mine_mask)
        ].copy()
        eia_mines_left_df = eia_mines_df[
            (invalid_points_mask) | (mismatched_mine_mask)
        ].copy()
        # print(f"eia mines w/ invalid points: {invalid_points.shape[0]:,}")
        print("eia mines with mismatched states : ", mismatched_state_mines.shape[0])
        print(f"eia mines with msha points: {eia_mines_msha_geo_gdf.shape[0]:,}")

        print(f"eia mines left: {eia_mines_left_df.shape[0]:,}")
        # %%

        # get the set of mines that still have no geo.
        # eia_mines_no_geo = pd.concat([mismatched_state_mines, eia_mines_left_df]).reset_index()
        # print(f"eia mines with no geo: {eia_mines_no_geo.shape[0]:,}")

        # try to get geom from the counties
        eia_mines_w_cnty_gdf = (
            county_gdf[["county_name", "state_name", "geometry", "state_code"]]
            .merge(
                eia_mines_left_df.reset_index(),
                on=["county_name", "state_name"],
                how="inner",
            )
            .set_index("MINE_ID")
        )

        eia_mines_left_df = eia_mines_left_df[
            ~eia_mines_left_df.index.isin(eia_mines_w_cnty_gdf.index)
        ]

        print(f"total EIA mines w/ county: {eia_mines_w_cnty_gdf.shape[0]:,}")
        print(f"total EIA mines w/ no geo: {eia_mines_left_df.shape[0]:,}")

        # this is used to get location data onto inventory mines and EIA mines via
        # the county it is listed in from EIA
        # %%
        eia_mines_gdf = pd.concat([eia_mines_msha_geo_gdf, eia_mines_w_cnty_gdf]).loc[
            :,
            [
                "mine_state",
                "state_name",
                "state_code",
                "county_name",
                "production",
                "geometry",
                "year",
                "mine_status",
            ],
        ]

        count_check = eia_mines_gdf.shape[0] == eia_mines_df.shape[0]
        if not count_check:
            raise ValueError(f"we are mssing mines!")
        print("\nnumber of NAs")
        display(eia_mines_gdf.isna().sum())
        print("\nare all geoms valid:")
        display(eia_mines_gdf.is_valid.all())
        eia_mines_gdf
        # %%

        print("how many mines are listed as active but have 0 production?")
        eia_mines_gdf.query(
            "mine_status == 'Active' & (production == 0)"
        ).mine_status.value_counts()

        print("how many mines are listed as NOT active but have production?")
        eia_mines_gdf.query(
            "mine_status != 'Active' & (production > 0)"
        ).mine_status.value_counts()

        sns.relplot(
            data=eia_mines_gdf,
            x="year",
            y="production",
            kind="line",
            hue="state_code",
            legend=True,
        )

        # %%
        def calc_prod_emi(data):
            # NOTE: although "Pennsylvania (Bituminous)" and "Pennsylvania (Anthracite)"
            # are listed in the original code, they are not calculated differently from
            # "other". It is not clear why they are listed separately.
            mine_state = data.name
            prod_coef_dict = {
                "Kentucky (East)": 61.4,
                "Kentucky (West)": 64.3,
                "West Virginia (Northern)": 138.4,
                "West Virginia (Southern)": 136.8,
            }

            if mine_state in list(prod_coef_dict.keys()):
                res = data * prod_coef_dict[mine_state]
            else:
                res = data
            return res

        eia_mines_gdf["weighted_prod"] = (
            eia_mines_gdf.groupby("mine_state")["production"]
            .transform(calc_prod_emi)
            .rename("weighted_production")
        )
        eia_mines_gdf["rel_emi"] = eia_mines_gdf.groupby(["year", "state_code"])[
            "weighted_prod"
        ].transform(normalize)

        # this checks to make sure the only data altered are only ones listed in the
        # fuction.
        display(
            eia_mines_gdf.query("production != weighted_prod")[
                "mine_state"
            ].value_counts()
        )
        # %%
        unique_eia_mines = (
            eia_mines_gdf.reset_index()[["MINE_ID", "state_name"]]
            .drop_duplicates()
            .sort_values("MINE_ID")
            .set_index("MINE_ID")
        )
        unique_eia_mines
        # %%
        duplicate_mine_ids = unique_eia_mines[
            unique_eia_mines.index.duplicated(keep=False)
        ]
        print(
            "print number of unique mines in multiple states: "
            f"{duplicate_mine_ids.index.value_counts().sort_values()}"
        )
        duplicate_mine_ids
        # %%
        # test_mine_id = 4609535
        if mine_type == "Surface":
            test_mine_id = 3609183
            test_mine_gdf = eia_mines_gdf[
                eia_mines_gdf.index == test_mine_id
            ].sort_values("year")
            display(test_mine_gdf)
            print(test_mine_gdf.year.nunique())
            ax = test_mine_gdf.plot(color="xkcd:lavender")
            msha_gdf[msha_gdf.index == test_mine_id].plot(
                ax=ax, color="xkcd:teal", markersize=50, zorder=10
            )
            ax.set_title(f"mine_id {test_mine_id}")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            state_gdf.boundary.plot(ax=ax, lw=1, color="xkcd:slate")
            county_gdf.boundary.plot(ax=ax, lw=0.25, color="xkcd:slate")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            sns.despine()
            plt.show()

        # %%
        # # now we fill in any states that are missing from the EIA data with MSHA data
        # # since MSHA do not provide any measure of relative emissions, we equally
        # # allocate hours to all mines in the state.
        # eia_supp_mines_list = []
        # for year in years:
        #     unique_states = eia_mines_gdf[eia_mines_gdf["year"] == year][
        #         "state_code"
        #     ].unique()
        #     missing_states = state_gdf[~state_gdf["state_code"].isin(unique_states)][
        #         "state_code"
        #     ].unique()
        #     # for state in missing_states:
        #     if missing_states.any():
        #         missing_mines = msha_gdf[
        #             (msha_gdf["state_code"].isin(missing_states))
        #             & (msha_gdf["CURRENT_MINE_STATUS"] == "Active")
        #         ].assign(year=year, emi=1)
        #         eia_supp_mines_list.append(missing_mines)
        # post_supp_mines_gdf = pd.concat(eia_supp_mines_list).assign(source="msha")
        # post_supp_mines_gdf["rel_emi"] = post_supp_mines_gdf.groupby(
        #     ["year", "state_code"]
        # )["emi"].transform(normalize)
        # post_supp_mines_gdf
        # post_proxy_gdf = pd.concat([eia_mines_gdf, post_supp_mines_gdf])[
        #     ["state_code", "year", "rel_emi", "geometry", "source"]
        # ]
        # %%
        post_proxy_gdf = eia_mines_gdf

        print("post mine count by year")
        display(post_proxy_gdf.reset_index().groupby("year")["MINE_ID"].nunique())
        post_all_close_1 = (
            post_proxy_gdf.groupby(["year", "state_code"])["rel_emi"]
            .sum()
            .apply(lambda x: np.isclose(x, 1))
        )
        if not post_all_close_1.all():
            print("post mines do not sum to 1")
            display(post_all_close_1[~post_all_close_1])
        # %%

        # if we are doing the underground mines, we need to get the mines from the
        # inventory workbook. We will join the inventory list of mines with MSHA to get
        # the geometry.

        ug_emi_col = "total liberated (mmcf)"

        inv_cols_list = [
            "state",
            "county",
            "basin",
        ] + [ug_emi_col]
        # if we are doing surface mines, we use the same EIA based proxy dataset for
        # both mining and post.
        if mine_type == "Underground":
            ug_inv_mines_list = []
            for year in tqdm(years, desc="getting underground inventory mines"):

                # read the mines list from the inventory
                sheet_name = f"UG-{year}"
                if year == 2012:
                    skip_rows = 2
                else:
                    skip_rows = 3
                ug_mine_df = (
                    pd.read_excel(
                        inventory_workbook_path,
                        sheet_name=sheet_name,
                        skiprows=skip_rows,
                    )
                    .iloc[: inv_ug_mine_count_by_year[year], :]
                    .rename(columns={"MSHA Mine ID": "MINE_ID"})
                    .astype({"MINE_ID": int})
                    .set_index("MINE_ID")
                    .rename(mapper=lambda x: x.lower().replace(str(f" {year}"), ""), axis=1)
                    .loc[:, inv_cols_list]
                    .rename(columns={ug_emi_col: "net_emi_tg"})
                    .assign(year=year)
                )
                ug_inv_mines_list.append(ug_mine_df)

            inv_mines_df = (
                pd.concat(ug_inv_mines_list)
                .query("net_emi_tg > 0")
                .rename(mapper=lambda x: x.lower().replace(" ", "_"), axis=1)
            )

            # This mine ID is not in MSHA and the county looks to me mislabeled.
            # It is in Somerset County, PA
            # https://en.wikipedia.org/wiki/Garrett,_Pennsylvania
            inv_mines_df.loc[
                (inv_mines_df["county"] == "Garret") & (inv_mines_df["state"] == "PA"),
                "county",
            ] = "Somerset"

            inv_mines_df = inv_mines_df.assign(
                county_name=lambda df: df["county"].str.casefold(),
            )

            inv_mines_msha_geo_gdf = msha_gdf[
                ["state_code", "geometry", "CURRENT_MINE_STATUS"]
            ].join(inv_mines_df, how="right")

            valid_geo_mask = inv_mines_msha_geo_gdf.is_valid & (
                ~inv_mines_msha_geo_gdf.is_empty
            )

            inv_no_geo_df = inv_mines_df[~valid_geo_mask]
            inv_mines_msha_geo_gdf = inv_mines_msha_geo_gdf[valid_geo_mask].copy()

            inv_no_geo_df = inv_no_geo_df.assign(
                state_code=lambda df: df["state"].str.strip().str.upper()
            )
            inv_w_cnty_gdf = (
                county_gdf[["county_name", "state_code", "geometry"]]
                .merge(
                    inv_no_geo_df.reset_index(),
                    on=["county_name", "state_code"],
                    how="right",
                )
                .set_index("MINE_ID")
            )
            inv_cnty_geo_mask = inv_w_cnty_gdf.is_valid & (~inv_w_cnty_gdf.is_empty)
            inv_w_cnty_gdf = inv_w_cnty_gdf[inv_cnty_geo_mask].copy()
            inv_no_geo_df = inv_w_cnty_gdf[~inv_cnty_geo_mask].copy()

            print(f"total unique mines:        {inv_mines_df.index.nunique():,}")
            print(f"mines w/ county geo:       {inv_w_cnty_gdf.index.nunique():,}")
            print(
                f"mines w/ geo:              {inv_mines_msha_geo_gdf.index.nunique():,}"
            )
            print(f"mines w/o geo:             {inv_no_geo_df.index.nunique():,}")
            inv_mines_gdf = pd.concat([inv_mines_msha_geo_gdf, inv_w_cnty_gdf])

            count_check = inv_mines_gdf.shape[0] == inv_mines_df.shape[0]
            if not count_check:
                raise ValueError(f"we are mssing mines!")

            inv_mines_gdf["rel_emi"] = inv_mines_gdf.groupby(["year", "state_code"])[
                "net_emi_tg"
            ].transform(normalize)

            # these totals should match the values listed in the EPA inventory workbook
            # sheet 'CM Emissions Summary', row 10 "Adj. Vent (VentUnadj/VentAdj %)"
            inv_mines_gdf.groupby("year")["net_emi_tg"].sum()
            inv_mines_gdf.groupby("year").size()
            coal_proxy_gdf = inv_mines_gdf
        else:
            coal_proxy_gdf = post_proxy_gdf

        # %%
        unique_mines_gdf = (
            post_proxy_gdf.reset_index()
            .drop_duplicates(subset="MINE_ID")
            .set_index("MINE_ID")
        )

        # reference plot of the mines
        _, ax = plt.subplots(dpi=300, figsize=(10, 10))
        state_gdf.boundary.plot(lw=0.5, color="xkcd:slate", ax=ax)
        unique_mines_gdf.plot(
            color="xkcd:teal",
            # "source",
            # categorical=True,
            # cmap="Set2",
            ax=ax,
            # legend=True,
            # legend_kwds={"fontsize": 8},
        )
        # leg = ax.get_legend()
        # leg.set_bbox_to_anchor((1.1, 0.75, 0.2, 0.2))
        ax.set(title=f"{mine_type} Coal Mines by State")
        sns.despine()
        plt.show()

        # %%
        coal_proxy_gdf.to_parquet(output_path_coal)
        post_proxy_gdf.to_parquet(output_path_coal_post)

        # %%


emi_dir_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/"
    "Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/emis/"
)


emi_file_list = [
    "coal_under_emi.csv",
    "coal_surf_emi.csv",
    "coal_post_under_emi.csv",
    "coal_post_surf_emi.csv",
]

for emi_file in emi_file_list:
    emi_path = emi_dir_path / emi_file
    missing_data = (
        pd.read_csv(emi_path, usecols=["state_code", "year", "ghgi_ch4_kt"])
        .query("ghgi_ch4_kt > 0")
        .query("state_code != 'AK'")
        .set_index(["state_code", "year"])
        .join(
            coal_proxy_gdf.groupby(["state_code", "year"]).size().rename("proxy_count")
        )
        .fillna(0)
        .query("proxy_count > 0")
        .all()["proxy_count"]
    )
    print(f"{missing_data} state/year pass QC: {emi_path.stem}")

emi_path = emi_dir_path / "coal_under_emi.csv"
emi_path = emi_dir_path / "coal_surf_emi.csv"
emi_path = emi_dir_path / "coal_post_under_emi.csv"
emi_path = emi_dir_path / "coal_post_surf_emi.csv"

emi_df = pd.read_csv(emi_path).query("ghgi_ch4_kt > 0").query("state_code != 'AK'")

# %%
fig, axs = plt.subplots(3, 4, figsize=(15, 15), dpi=300)
for ax, (year, mine_df) in zip(axs.ravel(), eia_mines_gdf.groupby("year")):
    # for ax, (year, mine_df) in zip(axs.ravel(), inv_mines_gdf.groupby("year")):
    year_emi_df = emi_df.query("year == @year").set_index("state_code")[["ghgi_ch4_kt"]]
    emi_states = year_emi_df.index.sort_values()
    year_emi_gdf = state_gdf.set_index("state_code").join(year_emi_df, how="right")
    proxy_states = mine_df["state_code"].drop_duplicates().sort_values()

    missing_states = emi_states[~emi_states.isin(proxy_states)]
    good_states = emi_states[emi_states.isin(proxy_states)]

    state_gdf.boundary.plot(color="xkcd:slate", lw=0.5, ax=ax)
    if not missing_states.empty:
        try:
            state_gdf[state_gdf.state_code.isin(missing_states)].plot(
                hatch="///",
                edgecolor="xkcd:scarlet",
                facecolor="none",
                ax=ax,
                zorder=10,
            )
        except:
            print("no")
    if not good_states.empty:
        state_gdf[state_gdf.state_code.isin(good_states)].plot(color="xkcd:teal", ax=ax)

    missing_emi_total = round(
        year_emi_df[year_emi_df.index.isin(missing_states)].sum().values[0], 2
    )
    print(f"there are {len(emi_states)} emi states")
    print(f"there are {len(proxy_states)} proxy states")
    print(f"number of mines {len(mine_df)}")
    display(missing_states)
    print()
    ax.set_axis_off()
    ax.set_title(f"{year} surface mines\n" f"missing {missing_emi_total} kt")
fig.tight_layout()
plt.show()
# %%
ax = eia_mines_gdf.query("mine_state == 'Kansas'").plot()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
state_gdf.boundary.plot(ax=ax, lw=0.5, color="xkcd:slate")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.show()
# %%
# Below I was trying to troubleshoot some of the production data differences between the
# v2 and v3 data. I was trying to compare the production data for underground mines.

# I do need to check that the wieghted production calculation is correct, but we don't
# have to worry that state totals align because we do not have to disaggregate from
# national to state like they did in v2.

# # %%
# tmp.rename(columns={"MSHA ID": "MINE_ID"}).set_index("MINE_ID").join(
#     msha_df, how="left"
# )
# # %%

# mines_v2_df = pd.read_parquet(
#     Path(
#         "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/"
#         "Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/tmp/"
#         "v2_ug_mines.parquet"
#     )
# ).rename(
#     {"Mine State": "mine_state", "Mine Type": "mine_type", "MSHA": "MINE_ID"}, axis=1
# )
# underground_mines_v2 = mines_v2_df.query("mine_type == 'Underground'").set_index(
#     "MINE_ID"
# )
# # %%
# v2_emissions = (
#     underground_mines_v2.filter(like="emi_")
#     .rename(columns=lambda x: x.replace("emi_", ""))
#     .melt(var_name="year", value_name="emissions", ignore_index=False)
#     .reset_index()
#     .astype({"year": int})
#     .set_index(["MINE_ID", "year"])
# )
# v2_emissions
# # %%
# v2_production = (
#     underground_mines_v2.filter(like="prod_")
#     .rename(columns=lambda x: x.replace("prod_", ""))
#     .melt(var_name="year", value_name="production", ignore_index=False)
#     .reset_index()
#     .astype({"year": int})
#     .set_index(["MINE_ID", "year"])
# )
# v2_production
# # %%
# v2_mines_long = v2_emissions.join(v2_production, how="outer")
# v2_mines_long
# # %%
# v2_mine_count_by_year = (
#     v2_mines_long.query("emissions > 0")
#     .reset_index()
#     .groupby("year")["MINE_ID"]
#     .nunique()
#     .rename("v2_mine_count")
# )
# v3_mine_count_by_year = (
#     inv_mines_df
#     .query("net_emi_tg > 0")
#     .groupby("year")
#     .size()
#     .rename("v3_mine_count")
# )
# v3_mine_count_by_year.to_frame().join(
#     v2_mine_count_by_year, lsuffix="_v3", rsuffix="_v2"
# ).assign(count_diff=lambda df: df["v3_mine_count"] - df["v2_mine_count"]).astype(
#     "Int64"
# )
# # %%
# state_prod_v2 = (
#     v2_mines_long.groupby(["mine_state", "year"])["production"]
#     .sum()
#     .rename("v2_prod")
#     .to_frame()
# )

# # %%
# state_prod_v3 = (
#     coal_proxy_gdf.groupby(["mine_state", "year"])["production"]
#     .sum()
#     .rename("v3_prod")
#     .to_frame()
# )
# # %%
# compare_state_prod = state_prod_v3.join(state_prod_v2, how="outer").assign(
#     diff=lambda df: df["v3_prod"] - df["v2_prod"]
# )
# compare_state_prod.query("diff != 0").reset_index().groupby(
#     ["mine_state", "year"]
# ).size().to_clipboard()
# # %%
# compare_state_prod.to_clipboard()
# # %%
# state_under_prod_v2 = (
#     pd.read_parquet(
#         Path(
#             "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/"
#             "Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/tmp/"
#             "state_under_prod.parquet"
#         )
#     )
#     .reset_index()
#     .rename(columns={"index": "state_name", "variable": "year"})
#     .merge(state_gdf.set_index("state_name")[["state_code"]], on="state_name")
#     .astype({"year": int})
#     .set_index(["year", "state_code"])
#     .drop(columns="state_name")
# )
# state_under_prod_v2

# # %%
# state_prod_v3_df = state_production.to_frame()
# state_prod_v3_df
# # %%


# compare_adj_prod = state_prod_v3_df.join(state_under_prod_v2, how="inner").assign(
#     diff=lambda df: df["weighted_production"] - df["value"]
# )
# compare_adj_prod.query("diff != 0").reset_index()["state_code"].value_counts()
# # %%
# compare_adj_prod.query("diff != 0").to_clipboard()

# # %%
# compare_mine_prod = v2_mines_long.join(inv_mines_df, rsuffix="_v3").assign(
#     prod_diff=lambda df: df["production"] - df["production_v3"],
# )
# compare_prod_not_0 = compare_mine_prod.query("prod_diff != 0")
# # %%
# compare_mine_prod.query("prod_diff != 0").reset_index()["MINE_ID"].nunique()
# # %%
# compare_mine_prod.reset_index()["MINE_ID"].nunique()
# # %%
# compare_mine_prod.loc[(504461, slice(None))]
# # %%
# compare_mine_prod.loc[(504461, slice(None))]

# # %%
# compare_mine_prod.query("(production != 0) and (prod_diff != 0)")
# %%
# active_underground_mines_df = msha_df.query(
#     "(CURRENT_MINE_TYPE == 'Underground') & CURRENT_MINE_STATUS == 'Active'"
# )
# active_underground_mines_gdf = gpd.GeoDataFrame(
#     active_underground_mines_df,
#     geometry=gpd.points_from_xy(
#         active_underground_mines_df.LONGITUDE, active_underground_mines_df.LATITUDE
#     ),
#     crs=4326,
# )

# all_emi_states = emi_df.state_code.drop_duplicates().sort_values()
# all_emi_states_gdf = state_gdf.set_index("state_code").loc[all_emi_states]

# active_underground_mines_gdf = active_underground_mines_gdf.sjoin(
#     state_gdf.set_index("state_code")[["geometry"]], how="left"
# )
# active_mine_states = active_underground_mines_gdf.groupby("state_code").size()
# active_mine_states_gdf = state_gdf.set_index("state_code").join(
#     active_mine_states.rename("mine_count")
# )
# ax = active_underground_mines_gdf.plot()
# active_mine_states_gdf.plot("mine_count", ax=ax, legend=True)
# all_emi_states_gdf.plot(ax=ax, color="xkcd:slate", zorder=-1)
# state_gdf.boundary.plot(ax=ax, zorder=-2)
# %%
