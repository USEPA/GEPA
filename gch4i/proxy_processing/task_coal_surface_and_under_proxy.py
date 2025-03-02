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

eia_cols = [
    "MSHA ID",
    "Mine Type",
    "Mine State",
    "Mine Status",
    "Production (short tons)",
]
status_filter = ["Active", "Active, men working, not producing"]


# Million cubic ft (mmcf) to Tg conversion factor - Source: EPA spreadsheet,
# 'CM Emissions Summary' cell C 40.
mmcf_to_Gg = 51921

EIA_dir_path = sector_data_dir_path / "coal/EIA"

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
) = param_dict["Surface"].values()


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

        # %%
        # load the MSHA mine data
        with ZipFile(msha_path) as z:
            with z.open("Mines.txt") as f:
                msha_df = pd.read_table(
                    f,
                    sep="|",
                    encoding="ISO-8859-1",
                    usecols=["MINE_ID", "LATITUDE", "LONGITUDE"],
                ).set_index("MINE_ID")
        # %%
        # for each year, we get the coal mines list from the inventory, read the
        # production data from that year from the production source, and and then join
        # the production and MSHA data to the yearly mine list. This is then saved to a
        # list and concatted together into the full mine list. We check that the lenth
        # of the inventory mine list is correct before and after the join. We outer join
        # with the production data so that we end up with a list of mines beyond what
        # the inventory provides when there are mines from the production data not in
        # the mine list. These mines are given 0 emissions.
        inv_mines_list = []
        eia_mines_list = []
        # add the EIA mine list by year, concat them together. norm the production by
        # state and year, then output for post mining.
        for year in years:
            print(f"processing {year} {mine_type} mines")

            eia_file_path = [x for x in eia_paths if str(year) in x.name][0]
            e_df = pd.read_excel(eia_file_path, skiprows=3, usecols=eia_cols)

            # rename some columns for easier use, get only the correct mine type
            e_df = (
                e_df.rename(
                    columns={
                        "MSHA ID": "MINE_ID",
                        "Production (short tons)": "production",
                        "Mine Type": "mine_type",
                        "Mine Status": "mine_status",
                    }
                )
                .drop_duplicates(subset=["MINE_ID"])
                .dropna(subset=["MINE_ID"])
                .set_index("MINE_ID")
                .join(msha_df, how="left")
                .query(
                    "(mine_type == @mine_type) & " "(mine_status.isin(@status_filter))"
                )
                .assign(year=year)
            )
            print(f"total EIA mines {e_df.index.nunique()}")
            e_df = e_df.dropna(subset=["LATITUDE", "LONGITUDE"])
            print(f"EIA mines with lat/lon {e_df.shape[0]}")

            # read the mines list from the inventory
            sheet_name = f"UG-{year}"
            if year == 2012:
                skip_rows = 2
            else:
                skip_rows = 3
            year_mine_df = pd.read_excel(
                inventory_workbook_path,
                sheet_name=sheet_name,
                skiprows=skip_rows,
            )

            # find the first row with all NaN values and drop everything after that
            # based on the structure of the inventory sheets, this will return the mines
            # list
            end_row = year_mine_df.index[year_mine_df.isna().all(axis=1)].min()
            if not np.isnan(end_row):
                year_mine_df = year_mine_df.iloc[:end_row, :]
                # print(inv_mine_df["MSHA Mine ID"].isna().sum())
            print(
                f"found {year_mine_df['MSHA Mine ID'].nunique()} " "mines in inventory"
            )
            print(f"missing MINE ID: {year_mine_df['MSHA Mine ID'].isna().sum()}")

            year_mine_df = (
                year_mine_df.rename(columns={"MSHA Mine ID": "MINE_ID"})
                .dropna(subset=["MINE_ID"])
                # MINE_ID, state, basin, total vent emis (mmcf/yr)
                .iloc[:, [0, 3, 6, 9]]
                .astype({"MINE_ID": int})
                .set_index("MINE_ID")
                # .join(e_df, how="left")
                .join(msha_df, how="left")
                # .dropna(subset=["LATITUDE", "LONGITUDE"])
                .assign(year=year)
                .rename(mapper=lambda x: x.replace(str(f" {year}"), ""), axis=1)
            )
            print(f"inventory mines with lat/lon: {year_mine_df.shape[0]}")
            # eia_matching_count = e_df.index.isin(year_mine_df.index).sum()
            # print(f"{year} EIA mines in inventory: {eia_matching_count}")
            # print(
            #     (
            #         f"{year} EIA mines not in inventory: "
            #         f"{e_df.shape[0] - eia_matching_count}"
            #     )
            # )
            print()
            eia_mines_list.append(e_df)
            inv_mines_list.append(year_mine_df)
            # %%
            # underground counts by year:
            # 2012: 117
            # 2013: 205
            # 2014: 178
            # 2015: 220
            # 2016: 163
            # 2017: 162
            # 2018: 164
            # 2019: 167
            # 2020: 231
            # 2021: 203
            # 2022: 209

        # %%

        # %%
        display(unique_mines_gdf["mine_state"].value_counts())
        display(unique_mines_gdf["mine_type"].value_counts())
        # reference plot of the mines
        _, ax = plt.subplots(dpi=300, figsize=(10, 10))
        state_gdf.boundary.plot(lw=0.5, color="xkcd:slate", ax=ax)
        unique_mines_gdf.plot(
            "mine_state",
            categorical=True,
            cmap="tab20",
            ax=ax,
            legend=True,
            legend_kwds={"fontsize": 8},
        )
        leg = ax.get_legend()
        ax.set(title="Underground Coal Mines by State")
        sns.despine()
        leg.set_bbox_to_anchor((1.1, 0.75, 0.2, 0.2))
        # %%

        post_mines_df = pd.concat(eia_mines_list).query("production > 0")
        post_mines_gdf = (
            gpd.GeoDataFrame(
                post_mines_df.drop(columns=["LATITUDE", "LONGITUDE"]),
                geometry=gpd.points_from_xy(
                    post_mines_df.LONGITUDE, post_mines_df.LATITUDE
                ),
                crs=4326,
            )
            .sjoin(state_gdf.set_index("state_code")[["geometry"]], how="left")
            .rename(mapper=lambda x: x.lower().replace(" ", "_"), axis=1)
        )
        print(f"Total post mines: {post_mines_gdf.index.nunique()}")
        print("post mine count by year:")
        display(post_mines_gdf.groupby("year").size())

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

        post_mines_gdf["weighted_prod"] = (
            post_mines_gdf.groupby("mine_state")["production"]
            .transform(calc_prod_emi)
            .rename("weighted_production")
        )
        # this checks to make sure the only data altered are only ones listed in the
        # fuction.
        display(
            post_mines_gdf.query("production != weighted_prod")[
                "mine_state"
            ].value_counts()
        )

        post_mines_gdf["rel_emi"] = post_mines_gdf.groupby(["year", "state_code"])[
            "weighted_prod"
        ].transform(normalize)
        print("post mine count by year")
        display(post_mines_gdf.reset_index().groupby("year")["MINE_ID"].nunique())
        post_all_close_1 = (
            post_mines_gdf.groupby(["year", "state_code"])["rel_emi"]
            .sum()
            .apply(lambda x: np.isclose(x, 1))
        )
        if not post_all_close_1.all():
            print("post mines do not sum to 1")
            display(post_all_close_1[~post_all_close_1])

        # %%
        # concat the yearly mines list together, calculate the net emi in tgs, rename
        # the columns so they are easier to work with, then fill any missing values with
        # 0. get mines that have production or emissions values (we won't use mines that
        # have 0 for both).
        inv_mines_df = (
            pd.concat(inv_mines_list)
            .reset_index()
            .set_index(["MINE_ID", "year"])
            .assign(net_emi_tg=lambda df: df["Total Vent Emis (mmcf/yr)"] / mmcf_to_Gg)
            .rename(mapper=lambda x: x.lower().replace(" ", "_"), axis=1)
            # .fillna({"production": 0, "net_emi_tg": 0})
            # .query("(production > 0) | (net_emi_tg > 0)")
        )
        inv_mines_df

        # create points from the lat/lons, spatial join with the states
        coal_proxy_gdf = (
            gpd.GeoDataFrame(
                inv_mines_df.drop(columns=["latitude", "longitude"]),
                geometry=gpd.points_from_xy(
                    inv_mines_df.longitude, inv_mines_df.latitude
                ),
                crs=4326,
            )
            .sjoin(state_gdf.set_index("state_code")[["geometry"]], how="inner")
            .rename(mapper=lambda x: x.lower().replace(" ", "_"), axis=1)
            .query("net_emi_tg > 0")
        )
        print(f"Total mines: {coal_proxy_gdf.index.nunique()}")
        print("mine count by year:")
        display(coal_proxy_gdf.groupby("year").size())

        # these totals should match the values listed in the EPA inventory workbook
        # sheet 'CM Emissions Summary', row 10 "Adj. Vent (VentUnadj/VentAdj %)"
        coal_proxy_gdf.groupby("year")["total_vent_emis_(mmcf/yr)"].sum()
        # They do!

        # %%
        # get a dataframe of just unique mines (no timeseries repeats)
        unique_mines_gdf = (
            coal_proxy_gdf.reset_index()
            .drop_duplicates(subset=["MINE_ID"], keep="first")
            .set_index("MINE_ID")[
                ["geometry", "state_code", "basin", "mine_state", "mine_type"]
            ]
        )
        print(f"Unique mines: {unique_mines_gdf.index.nunique()}")

        coal_proxy_gdf["rel_emi"] = coal_proxy_gdf.groupby(["year", "state_code"])[
            "net_emi_tg"
        ].transform(
            normalize
        )  # normalize to sum to 1
        print(f"{mine_type} mine count by year")
        display(coal_proxy_gdf.reset_index().groupby("year")["MINE_ID"].nunique())
        under_all_close_1 = (
            coal_proxy_gdf.groupby(["year", "state_code"])["rel_emi"]
            .sum()
            .apply(lambda x: np.isclose(x, 1))
        )
        if not under_all_close_1.all():
            print("post mines do not sum to 1")
            display(under_all_close_1[~under_all_close_1])

        # %%
        fig, ax = plt.subplots(dpi=300, figsize=(10, 10))
        state_gdf.boundary.plot(lw=0.5, color="xkcd:slate", ax=ax)
        post_mines_gdf.query("year == 2022").plot(
            "rel_emi",
            cmap="Greens",
            ax=ax,
            legend=True,
            legend_kwds={"shrink": 0.3},
        )
        coal_proxy_gdf.query("year == 2022").plot("rel_emi", cmap="Purples", ax=ax)
        ax.set(title=f"{mine_type} Coal Mine Relative Emissions by State for 2022")
        sns.despine()
        plt.show()
        # %%
        coal_proxy_gdf.to_parquet(output_path_coal)
        post_mines_gdf.to_parquet(output_path_coal_post)

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
# # %%
