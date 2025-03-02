"""
Name:                   test_emi_proxy_mapping.py
Date Last Modified:     2025-01-30
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This is a prototype file that would preceed the final gridding.
                        It is intended to check the data files for the proxies and
                        emissions to ensure that they are formatted correctly and
                        contain the necessary columns for the gridding process. This
                        file will output a csv file that will be used to guide the
                        gridding process.

                        This file could also act as the final QC step before the
                        gridding process.
                        TODO: continue to add on checks for emi files and proxy files
                        especially around the use of time [year, month] and spatial
                        [state, county] dimensions and alignment between the emi and
                        proxy pairs.
                        TODO: finish the checking of netcdf files. I had just begun
                        that process when I had to stop working on this file.
                        TODO: below the to_csv that writes out the new emi proxy map
                        file, I began to work on the QC type steps that would help
                        identify files that needed to be updated. Almost always this is
                        going to be proxy files.
Input Files:            - all emi and proxy files
Output Files:           - emi_proxy_mapping_output.csv
"""

import warnings
from pathlib import Path

import duckdb
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from gch4i.config import V3_DATA_PATH, emi_data_dir_path, proxy_data_dir_path


REL_EMI_COL_LIST = ["rel_emi", "ch4", "emis_kt", "ch4_flux"]


def make_has_cols_bool(in_df):
    has_cols = in_df.filter(like="has").columns
    in_df = in_df.astype({col: bool for col in has_cols})
    return in_df


def check_emi_file(input_path):
    in_df = duckdb.execute(f"SELECT * FROM '{input_path}' LIMIT 0").df()
    col_list = list(in_df.columns.str.lower())
    emi_has_state_col = "state_code" in col_list
    # emi_has_county_col = "county" in col_list
    emi_has_county_col = "fips" in col_list
    emi_has_year_col = "year" in col_list
    emi_has_month_col = "month" in col_list
    emi_has_emi_col = "ghgi_ch4_kt" in col_list

    if emi_has_county_col:
        emi_geo_level = "county"
    elif emi_has_state_col:
        emi_geo_level = "state"
    else:
        emi_geo_level = "national"

    if emi_has_month_col:
        emi_time_step = "monthly"
    else:
        emi_time_step = "annual"

    emi_res_dict = dict(
        emi_geo_level=emi_geo_level,
        emi_time_step=emi_time_step,
        emi_has_state_col=emi_has_state_col,
        emi_has_fips_col=emi_has_county_col,
        emi_has_county_col=emi_has_county_col,
        emi_has_year_col=emi_has_year_col,
        emi_has_month_col=emi_has_month_col,
        emi_has_emi_col=emi_has_emi_col,
    )

    return emi_res_dict, col_list


def check_proxy_file(input_path):
    in_gdf = duckdb.execute(f"SELECT * FROM '{input_path}' LIMIT 0").df()
    col_list = list(in_gdf.columns.str.lower())

    proxy_has_state_col = "state_code" in col_list
    proxy_has_county_col = "county" in col_list
    proxy_has_year_col = "year" in col_list
    proxy_has_year_month_col = "year_month" in col_list
    proxy_has_month_col = "month" in col_list
    proxy_has_geom_col = "geometry" in col_list
    proxy_has_rel_emi_col = any(x in col_list for x in REL_EMI_COL_LIST)
    if proxy_has_rel_emi_col:
        proxy_rel_emi_col = [x for x in REL_EMI_COL_LIST if x in col_list][0]
    else:
        proxy_rel_emi_col = None

    if proxy_has_year_month_col | proxy_has_month_col:
        proxy_time_step = "monthly"
    else:
        proxy_time_step = "annual"

    if proxy_has_county_col:
        proxy_geo_level = "county"
    elif proxy_has_state_col:
        proxy_geo_level = "state"
    else:
        proxy_geo_level = "national"

    res_dict = dict(
        proxy_time_step=proxy_time_step,
        proxy_geo_level=proxy_geo_level,
        proxy_has_file=True,
        proxy_has_state_col=proxy_has_state_col,
        proxy_has_county_col=proxy_has_county_col,
        proxy_has_year_col=proxy_has_year_col,
        proxy_has_month_col=proxy_has_month_col,
        proxy_has_year_month_col=proxy_has_year_month_col,
        proxy_has_geom_col=proxy_has_geom_col,
        proxy_has_rel_emi_col=proxy_has_rel_emi_col,
        proxy_rel_emi_col=proxy_rel_emi_col,
        file_type="parquet",
    )

    # return pd.DataFrame.from_dict({proxy_name: res_dict}, orient="index")
    return res_dict, col_list


def check_proxy_nc(input_path):
    xr_ds = xr.open_dataset(input_path, chunks="auto")
    coords = list(xr_ds.coords.keys())
    data_vars = list(xr_ds.data_vars.keys())
    data_vars.remove("spatial_ref")
    if data_vars:
        if len(data_vars) > 1:
            warnings.Warn(
                f"More than one data variable in the netcdf file: {data_vars}"
            )
        proxy_has_rel_emi_col = True
        proxy_rel_emi_col = data_vars[0]
    else:
        proxy_has_rel_emi_col = False
        proxy_rel_emi_col = None

    proxy_has_state_col = "statefp" in coords
    proxy_has_county_col = "geoid" in coords
    proxy_has_year_col = "year" in coords
    proxy_has_geom_col = all(x in coords for x in ["x", "y"])

    if proxy_has_year_col:
        proxy_time_step = "annual"
    else:
        proxy_time_step = "monthly"

    if proxy_has_county_col:
        proxy_geo_level = "county"
    else:
        proxy_geo_level = "state"

    res_dict = dict(
        proxy_time_step=proxy_time_step,
        proxy_geo_level=proxy_geo_level,
        proxy_has_file=True,
        proxy_has_state_col=proxy_has_state_col,
        proxy_has_county_col=proxy_has_county_col,
        proxy_has_year_col=proxy_has_year_col,
        proxy_has_month_col=False,
        proxy_has_year_month_col=False,
        proxy_has_geom_col=proxy_has_geom_col,
        proxy_has_rel_emi_col=proxy_has_rel_emi_col,
        proxy_rel_emi_col=proxy_rel_emi_col,
        file_type="netcdf",
    )
    return res_dict


def get_gridding_mapping_df(
    mapping_file_path: Path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx",
    save_file: bool = True,
):

    guide_sheet = pd.read_excel(mapping_file_path, sheet_name="emi_proxy_mapping")

    emi_proxy_mapping = guide_sheet[
        ["gch4i_name", "emi_id", "proxy_id"]
    ].drop_duplicates()
    emi_proxy_mapping

    n_unique_proxies = emi_proxy_mapping.proxy_id.nunique()
    n_unique_emis = emi_proxy_mapping.emi_id.nunique()

    all_proxy_cols = {}
    proxy_results = {}
    for data in tqdm(
        guide_sheet.drop_duplicates("proxy_id").sort_values("proxy_id").itertuples(),
        desc="Checking proxy files",
        total=n_unique_proxies,
    ):
        proxy_name = data.proxy_id
        proxy_paths = list(proxy_data_dir_path.glob(f"{proxy_name}.*"))
        if proxy_paths:
            proxy_path = proxy_paths[0]
            if proxy_path.suffix == ".parquet":
                proxy_path = list(proxy_paths)[0]
                proxy_name = proxy_path.stem
                check_proxy, proxy_cols = check_proxy_file(proxy_path)
                proxy_results[proxy_name] = check_proxy
                all_proxy_cols[proxy_name] = proxy_cols
            elif proxy_path.suffix == ".nc":
                proxy_path = list(proxy_paths)[0]
                check_proxy = check_proxy_nc(proxy_path)
                proxy_results[proxy_name] = check_proxy
                # all_proxy_cols[proxy_name] = proxy_cols

        else:
            print(f"{proxy_name} not found")

    all_emi_cols = {}
    emi_results = {}
    for data in tqdm(
        guide_sheet.drop_duplicates("emi_id").sort_values("emi_id").itertuples(),
        desc="Checking emi files",
        total=n_unique_emis,
    ):
        emi_name = data.emi_id
        emi_paths = list(emi_data_dir_path.glob(f"{emi_name}.csv"))
        if not emi_paths:
            print(f"{emi_name} not found")
            continue
        emi_path = emi_paths[0]
        check_emi, col_list = check_emi_file(emi_path)
        emi_results[emi_name] = check_emi
        all_emi_cols[emi_name] = col_list

    proxy_result_df = pd.DataFrame.from_dict(proxy_results, orient="index")
    proxy_result_df = make_has_cols_bool(proxy_result_df)

    emi_result_df = pd.DataFrame.from_dict(emi_results, orient="index")
    emi_result_df = make_has_cols_bool(emi_result_df)

    mapping_df = (
        emi_proxy_mapping.merge(
            emi_result_df.drop(columns=["emi_has_emi_col"]),
            left_on="emi_id",
            right_index=True,
            how="left",
        )
        .merge(
            proxy_result_df,
            left_on="proxy_id",
            right_index=True,
            how="left",
        )
        .fillna({"proxy_has_file": False})
    )
    if save_file:
        mapping_df.to_csv(
            V3_DATA_PATH.parents[1] / "emi_proxy_mapping_output.csv", index=False
        )

    # currently we are missing some proxies, so we drop any gridding group that does not
    # have all the proxies ready.
    # groups_ready_for_gridding = mapping_df.groupby("gch4i_name")["proxy_has_file"].all()
    # groups_ready_for_gridding = groups_ready_for_gridding[groups_ready_for_gridding]
    # filter the mapping df to only include the groups that are ready for gridding.
    # mapping_df = mapping_df[mapping_df.gch4i_name.isin(groups_ready_for_gridding.index)]

    return mapping_df


# %%
# proxy_cols_df = pd.DataFrame.from_dict(all_proxy_cols, orient="index")
# proxy_cols_df
# # %%
# flat_proxy_col_list = [x for xs in all_proxy_cols.values() for x in xs]
# set(flat_proxy_col_list)

# # %%
# emi_cols_df = pd.DataFrame.from_dict(all_emi_cols, orient="index")

# all_emis_have_state = emi_result_df.emi_has_state_col.all()
# all_emis_have_year = emi_result_df.emi_has_year_col.all()
# all_emis_have_emi = emi_result_df.emi_has_emi_col.all()

# if not all_emis_have_state:
#     missing_state_emis = emi_result_df[-emi_result_df.emi_has_state_col]
#     display(missing_state_emis)
#     for data in missing_state_emis.itertuples():
#         emi_name = data.Index
#         emi_path = list(emi_data_dir_path.glob(f"{emi_name}.csv"))[0]
#         in_df = duckdb.execute(f"SELECT * FROM '{emi_path}' LIMIT 5").df()
#         display(in_df.head())
#         # col_list = list(in_df.columns)
#         # print(f"{emi_name}: {col_list}")

# if not all_emis_have_year:
#     missing_year_emis = emi_result_df[-emi_result_df.emi_has_year_col]
#     display(missing_year_emis)
#     for data in missing_year_emis.itertuples():
#         emi_name = data.Index
#         emi_path = list(emi_data_dir_path.glob(f"*{emi_name}*"))[0]
#         in_df = duckdb.execute(f"SELECT * FROM '{emi_path}' LIMIT 5").df()
#         display(in_df.head())
#         # col_list = list(in_df.columns)
#         # print(f"{emi_name}: {col_list}")
# else:
#     print("All emis have year column")

# if not all_emis_have_emi:
#     missing_emi_emis = emi_result_df[-emi_result_df.emi_has_emi_col]
#     display(missing_emi_emis)
#     for data in missing_emi_emis.itertuples():
#         emi_name = data.Index
#         emi_path = list(emi_data_dir_path.glob(f"*{emi_name}*"))[0]
#         in_df = duckdb.execute(f"SELECT * FROM '{emi_path}' LIMIT 5").df()
#         display(in_df.head())
#         # col_list = list(in_df.columns)
#         # print(f"{emi_name}: {col_list}")

# # proxy_result_df[proxy_result_df.index.duplicated()]

# # for col in proxy_result_df.filter(like="has").columns:
# #     if not proxy_result_df[col].all():
# #         print(f"Missing {col}")
# #         display(proxy_result_df[~proxy_result_df[col]])
# #         for data in proxy_result_df[~proxy_result_df[col]].itertuples():
# #             proxy_name = data.Index
# #             print(proxy_name)
# #             proxy_path = list(proxy_data_dir_path.glob(f"{proxy_name}.*"))[0]
# #             proxy_df = duckdb.execute(f"SELECT * FROM '{proxy_path}' LIMIT 5").df()
# #             display(proxy_df.head())
# #             print()
# #             # proxy_cols = list(proxy_df.columns)
# #             # print(f"{proxy_name}: {proxy_cols}")
# #     print()


# # %%
