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
from gch4i.utils import download_url


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
        for year in years:

            eia_file_path = [x for x in eia_paths if str(year) in x.name][0]
            e_df = pd.read_excel(eia_file_path, skiprows=3, usecols=eia_cols)

          #EEM: e_df is a list of both underground and surface mines, right? I think we want to have separate mines lists for underground and surface

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
                .set_index("MINE_ID")
                .query(
                    "(mine_type == @mine_type) & " "(mine_status.isin(@status_filter))"
                )
            )

            # read the mines list from the inventory
          #EEM: the inventory workbooks only includes a list of underground mines, right?
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
                f"{year} found {year_mine_df['MSHA Mine ID'].nunique()} "
                "mines in inventory"
            )
            print(
                f"{year} missing MINE ID: {year_mine_df['MSHA Mine ID'].isna().sum()}"
            )

          # EEM: does this join limit the year_mine_df list to those from the inventory only, but add the msha and eia data to that list?
          ## What I'm asking is whether this list of mines includes all surface and underground mines, or just underground mines. 
            year_mine_df = (
                year_mine_df.rename(columns={"MSHA Mine ID": "MINE_ID"})
                .dropna(subset=["MINE_ID"])
                # MINE_ID, state, basin, total vent emis (mmcf/yr)
                .iloc[:, [0, 3, 6, 9]]
                .astype({"MINE_ID": int})
                .set_index("MINE_ID")
                .join(e_df, how="left")
                .join(msha_df, how="left")
                # .dropna(subset=["LATITUDE", "LONGITUDE"])
                .assign(year=year)
                .rename(mapper=lambda x: x.replace(str(f" {year}"), ""), axis=1)
            )
            print(f"{year} found {year_mine_df.shape[0]} mines with prod and loc")
            eia_matching_count = e_df.index.isin(year_mine_df.index).sum()
            print(f"{year} EIA mines in inventory: {eia_matching_count}")
            print(
                (
                    f"{year} EIA mines not in inventory: "
                    f"{e_df.shape[0] - eia_matching_count}"
                )
            )
            print()

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
        # concat the yearly mines list together, calculate the net emi in tgs, rename
        # the columns so they are easier to work with, then fill any missing values with
        # 0. get mines that have production or emissions values (we won't use mines that
        # have 0 for both).
        inv_mines_df = (
            pd.concat(inv_mines_list)
            .reset_index()
            .set_index(["MINE_ID", "year"])
            .assign(net_emi_tg=lambda df: df["Total Vent Emis (mmcf/yr)"] / mmcf_to_Gg) #EEM: the UG liberated emissions for each state are actually the sum of the 
          # Total Vent Emis column + Total Drainage (mmcf) columns. So that should be added here. Also, in the spreadsheets, these Total Vent and Total Drainage
          # columns also have the year in the column headers (e.g., Total Drainage 2020 (mmcf)", so please confirm that those years are not messing up the column assignment in this step
          # Also confirming that we do not need to add the relative EF weights for underground mines because the proxy is the relative emission, not the relative production
            .rename(mapper=lambda x: x.lower().replace(" ", "_"), axis=1)
            # .fillna({"production": 0, "net_emi_tg": 0})
            # .query("(production > 0) | (net_emi_tg > 0)")
        )
        inv_mines_df

      # EEM: note for future improvement, the GHGI workbook actually includes the avoided emissions by mine that are used to calculate the underground recovered and used emissions. 
      ## Currently, when processing the GHGI emissions, we consider the sum of liberated + recovered and used emissions at the state level, but we could update the methods
      ## in the future to calculate the net liberated + recovered & used emissions at the mine level instead. This is only relevant to the underground mine emissions. 
      
        # %%
        # create points from the lat/lons, spatial join with the states
      # EEM: we are missing a step here from v2 where the MSHA lat/lons for underground and surface mines were corrected based on a previous comparison with 
      # Google map searches. The files with the corrected locations are already in the data input folder. The example code from v2 is in Step 2.4.3: https://github.com/USEPA/GEPA/blob/v1.0/code/GEPA_Coal/1B1a_Coal.ipynb
      # This is also where we could make the mine location correction of the San Juan mine in NM that a user pointed out previously. 
        inv_mines_gdf = gpd.GeoDataFrame(
            inv_mines_df.drop(columns=["latitude", "longitude"]),
            geometry=gpd.points_from_xy(inv_mines_df.longitude, inv_mines_df.latitude),
            crs=4326,
        ).sjoin(state_gdf.set_index("state_code")[["geometry"]], how="inner")
        print(f"Total mines: {inv_mines_gdf.index.nunique()}")
        print("mine count by year:")
        display(inv_mines_gdf.groupby("year").size())
        # %%
        # how many mines have production data but no emissions?
        inv_mines_gdf.query(
            "(production > 0) & (`total_vent_emis_(mmcf/yr)` <= 0)"
        ).groupby("year").size()
        # %%
        # how many mines have emissions data but no production?
        inv_mines_gdf.query(
            "(`total_vent_emis_(mmcf/yr)` > 0) & (production.isna())"
        ).groupby("year").size()
        # %%
        inv_mines_gdf.groupby("year").apply(lambda x: x["production"].isna().sum())

        # %%
        # these totals should match the values listed in the EPA inventory workbook
        # sheet 'CM Emissions Summary', row 10 "Adj. Vent (VentUnadj/VentAdj %)"
        inv_mines_gdf.groupby("year")["total_vent_emis_(mmcf/yr)"].sum()
        # They do!
        # %%
      #EEM: is the below comment old? If so, please delete
        # XXX: where do I find in the inventory workbook the total production for the
        # year to validate these values?
        inv_mines_gdf.groupby("year")["net_emi_tg"].sum()

        # %%
        # get a dataframe of just unique mines (no timeseries repeats)
        unique_mines_gdf = (
            inv_mines_gdf.reset_index()
            .drop_duplicates(subset=["MINE_ID"], keep="first")
            .set_index("MINE_ID")[
                ["geometry", "state_code", "basin", "mine_state", "mine_type"]
            ]
        )
        print(f"Unique mines: {unique_mines_gdf.index.nunique()}")

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
        # TODO: if EPA approves, split the mines into post and regular here. then
        # proceed with the weighted production calculation and normalization for each
        # of them.
      #EEM: is this a list of the underground mines only or both surface and underground?
        post_proxy_gdf = inv_mines_gdf.query("production > 0").copy()

      #EEM: these factors are applied because there are multiple basins within KY and WV that have different EF's, 
      # and we want to add that relative weighting to the mines in each basin. We don't need to weight by the EF's
      # for the other states because all the mines in the state have the same EF and therefore, the only difference
      # between the mines in those states are the relative production differences. At some point the mines in PA had
    # different EF's, but that is no longer the case, so PA can be treated like all the other states. 
        # HOWEVER, we're getting vastly different values than v2, so something is wrong.

# EEM: these are the post-mining, underground EF's for the different basins in KY and WV, from the 'Post U E' tab in the GHGI workbook
        def calc_prod_emi_und(data):
            mine_state = data.name
            print(mine_state)
            prod_coef_dict = {
                "Kentucky (East)": 61.4,
                "Kentucky (West)": 64.3,
                "West Virginia (Northern)": 138.4,
                "West Virginia (Southern)": 136.8,
            }

            if mine_state in list(prod_coef_dict.keys()):
                res = data["production"] * prod_coef_dict[mine_state]
            else:
                res = data["production"]
            return res

        post_proxy_gdf["weighted_prod"] = (
            post_proxy_gdf.groupby(["mine_state"])
            .apply(calc_prod_emi_und, include_groups=False)
            .rename("weighted_production")
            .droplevel([0])
        )

        post_proxy_gdf["rel_emi"] = post_proxy_gdf.groupby(["year", "state_code"])[
            "weighted_prod"
        ].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )  # normalize to sum to 1
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
          
        coal_proxy_gdf = inv_mines_gdf.query("net_emi_tg > 0").copy()

        coal_proxy_gdf["rel_emi"] = coal_proxy_gdf.groupby(["year", "state_code"])[
            "net_emi_tg"
        ].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
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
        post_proxy_gdf.query("year == 2022").plot(
            "rel_emi",
            cmap="Spectral",
            ax=ax,
            legend=True,
            legend_kwds={"shrink": 0.3},
        )
        coal_proxy_gdf.query("year == 2022").plot("rel_emi", cmap="Spectral", ax=ax)
        ax.set(title=f"{mine_type} Coal Mine Relative Emissions by State for 2022")
        sns.despine()
        plt.show()

        coal_proxy_gdf.to_parquet(output_path_coal)
        post_proxy_gdf.to_parquet(output_path_coal_post)

        # %%
        #EEM we also need separate coal proxies for surface mines and surface post mining
        # this list of mines is from EIA (already processed as e_df, but filtered to only include 'Surface' mines)
        # Both active mines and post-mining activities for surface mines will be based on relative mine production levels
        # therefore, similar to the und post_proxy above, we need to correct the KY and WV production values with the relative EF weights (from the 'Surface & Post SU' tab in the GHGI workbook)
        def calc_prod_emi_surf(data):
            mine_state = data.name
            print(mine_state)
            prod_coef_dict = {
                "Kentucky (East)": 24.9,
                "Kentucky (West)": 34.3,
                "West Virginia (Northern)": 59.5,
                "West Virginia (Southern)": 24.9,
            }

            if mine_state in list(prod_coef_dict.keys()):
                res = data["production"] * prod_coef_dict[mine_state]
            else:
                res = data["production"]
            return res

#next, calculate the coal_surf_proxy using the weighted relative production values for each mine and apply to both surface and post mining surface emissions

#EEM: can we delete the comment below?

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
#     inv_mines_gdf.groupby(["mine_state", "year"])["production"]
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
