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

import calendar
import logging
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import duckdb
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import seaborn as sns
import xarray as xr
from IPython.display import display
from joblib import Parallel, delayed
from matplotlib.colors import TwoSlopeNorm
from rasterio.features import rasterize
from tqdm.auto import tqdm

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    global_data_dir_path,
    logging_dir,
    max_year,
    min_year,
    proxy_data_dir_path,
    status_db_path,
    years,
)
from gch4i.utils import (
    Avogadro,
    GEPA_spatial_profile,
    Molarch4,
    get_cell_gdf,
    load_area_matrix,
    normalize,
)

logger = logging.getLogger(__name__)


REL_EMI_COL_LIST = ["rel_emi", "ch4", "emis_kt", "ch4_flux"]


class GriddingInfo:
    def __init__(self, update_mapping=False, save_file=False):
        self.update_mapping: bool = update_mapping
        self.save_file: bool = save_file
        self.v2_data_path: Path = (
            V3_DATA_PATH.parents[1] / "v2_v3_comparison_crosswalk.csv"
        )
        self.data_guide_path: Path = (
            V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
        )
        self.emi_proxy_info_path: Path = (
            V3_DATA_PATH.parents[1] / "emi_proxy_mapping_output.csv"
        )
        self.status_db_path: Path = logging_dir / "gridding_status.db"
        self.guide_sheet = pd.read_excel(
            self.data_guide_path, sheet_name="emi_proxy_mapping"
        )
        self.v2_df = pd.read_csv(self.v2_data_path).rename(
            columns={"v3_gch4i_name": "gch4i_name"}
        )
        self.get_mapping_df()
        self.get_ready_pairs()
        self.get_ready_groups()

    def get_mapping_df(self):

        if self.update_mapping:
            emi_proxy_mapping = self.guide_sheet[
                ["gch4i_name", "emi_id", "proxy_id"]
            ].drop_duplicates()
            emi_proxy_mapping

            n_unique_proxies = emi_proxy_mapping.proxy_id.nunique()
            n_unique_emis = emi_proxy_mapping.emi_id.nunique()

            all_proxy_cols = {}
            proxy_results = {}
            for data in tqdm(
                self.guide_sheet.drop_duplicates("proxy_id")
                .sort_values("proxy_id")
                .itertuples(),
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
                        check_proxy, proxy_cols = self.check_proxy_file(proxy_path)
                        proxy_results[proxy_name] = check_proxy
                        all_proxy_cols[proxy_name] = proxy_cols
                    elif proxy_path.suffix == ".nc":
                        proxy_path = list(proxy_paths)[0]
                        check_proxy = self.check_proxy_nc(proxy_path)
                        proxy_results[proxy_name] = check_proxy
                        # all_proxy_cols[proxy_name] = proxy_cols

                else:
                    print(f"{proxy_name} not found")

            all_emi_cols = {}
            emi_results = {}
            for data in tqdm(
                self.guide_sheet.drop_duplicates("emi_id")
                .sort_values("emi_id")
                .itertuples(),
                desc="Checking emi files",
                total=n_unique_emis,
            ):
                emi_name = data.emi_id
                emi_paths = list(emi_data_dir_path.glob(f"{emi_name}.csv"))
                if not emi_paths:
                    print(f"{emi_name} not found")
                    continue
                emi_path = emi_paths[0]
                check_emi, col_list = self.check_emi_file(emi_path)
                emi_results[emi_name] = check_emi
                all_emi_cols[emi_name] = col_list

            proxy_result_df = pd.DataFrame.from_dict(proxy_results, orient="index")
            proxy_result_df = self.make_has_cols_bool(proxy_result_df)

            emi_result_df = pd.DataFrame.from_dict(emi_results, orient="index")
            emi_result_df = self.make_has_cols_bool(emi_result_df)

            self.mapping_df = (
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
            if self.save_file:
                self.mapping_df.to_csv(self.emi_proxy_info_path, index=False)
        else:
            if not self.emi_proxy_info_path.exists():
                raise FileNotFoundError(
                    "The emi_proxy_info_path file does not exist. "
                    "Please run the update_mapping option to create it."
                )

            self.mapping_df = pd.read_csv(self.emi_proxy_info_path)

    def make_has_cols_bool(self, in_df):
        has_cols = in_df.filter(like="has").columns
        in_df = in_df.astype({col: bool for col in has_cols})
        return in_df

    def check_emi_file(self, input_path):
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

    def check_proxy_file(self, input_path):
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

    def check_proxy_nc(self, input_path):
        xr_ds = xr.open_dataset(input_path, chunks="auto")
        coords = list(xr_ds.coords.keys())
        data_vars = list(xr_ds.data_vars.keys())
        data_vars.remove("spatial_ref")
        if data_vars:
            if len(data_vars) > 1:
                warnings.warn(
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
        proxy_has_year_month_col = "year_month" in coords
        proxy_has_geom_col = all(x in coords for x in ["x", "y"])

        if proxy_has_year_month_col:
            proxy_time_step = "monthly"
        else:
            proxy_time_step = "annual"

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

    def display_time_geo_summary(self):
        display(
            (
                self.mapping_df.groupby(
                    [
                        "file_type",
                        "emi_time_step",
                        "proxy_time_step",
                        "emi_geo_level",
                        "proxy_geo_level",
                    ]
                )
                .size()
                .rename("pair_count")
                .reset_index()
            )
        )

    def get_status_table(self, save=True):
        # get a numan readable version of the status database
        conn = sqlite3.connect(self.status_db_path)
        self.status_df = pd.read_sql_query("SELECT * FROM gridding_status", conn)
        conn.close()
        if save:
            the_date = datetime.now().strftime("%Y-%m-%d")
            self.status_df.to_csv(
                logging_dir / f"gridding_status_{the_date}.csv", index=False
            )

    def get_ready_pairs(self):
        # get the emi/proxy pairs that are ready for gridding
        self.get_status_table(save=False)

        self.pairs_ready_for_gridding_df = self.mapping_df.drop_duplicates(
            subset=["gch4i_name", "emi_id", "proxy_id"]
        )
        self.pairs_ready_for_gridding_df = self.pairs_ready_for_gridding_df.merge(
            self.status_df, on=["gch4i_name", "emi_id", "proxy_id"]
        )

        # if SKIP:
        #     self.pairs_ready_for_gridding_df = self.pairs_ready_for_gridding_df[
        #         ~self.pairs_ready_for_gridding_df["status"].isin(SKIP_THESE)
        #     ]

    def display_group_emi_proxy_statuses(self):
        pass

    def get_ready_groups(self):
        # get the status of each gridding group
        self.get_status_table(save=False)
        self.group_ready_status = (
            self.status_df.groupby("gch4i_name")["status"]
            .apply(lambda x: x.eq("complete").all())
            .to_frame()
        )

        # filter the emi/proxy data to only those groups that are ready for gridding
        # NOTE: not all v3 gridding groups have a v2 product
        self.ready_groups_df = (
            self.group_ready_status[self.group_ready_status["status"] == True]
            .join(self.v2_df.set_index("gch4i_name"))
            .fillna({"v2_key": ""})
            .astype({"v2_key": str})
        ).merge(self.mapping_df, on="gch4i_name", how="left")

    def display_all_group_statuses(self):
        # display the progress of the gridding groups
        print("percent of gridding groups ready")
        display(
            self.group_ready_status["status"]
            .value_counts(normalize=True)
            .multiply(100)
            .round(2)
        )

    def display_all_pair_statuses(self):
        # display the progress of the emi/proxy pairs
        print("percent of emi/proxy pairs by status")
        display(
            self.status_df["status"].value_counts(normalize=True).multiply(100).round(2)
        )


class BaseGridder(object):
    def __init__(self):
        self.state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
        self.county_geo_path = global_data_dir_path / "tl_2020_us_county.zip"
        self.gepa_profile = GEPA_spatial_profile()
        self.get_geo_filter()

    def get_state_gdf(self):
        self.state_gdf = (
            gpd.read_file(self.state_geo_path)
            .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
            .rename(columns=str.lower)
            .rename(columns={"stusps": "state_code", "name": "state_name"})
            .astype({"statefp": int})
            # get only lower 48 + DC
            .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
            .rename(columns={"statefp": "fips"})
            .to_crs(4326)
        )

    def get_county_gdf(self):
        self.county_gdf = (
            gpd.read_file(self.county_geo_path)
            .loc[:, ["NAME", "STATEFP", "COUNTYFP", "geometry"]]
            .rename(columns=str.lower)
            .rename(columns={"name": "county_name"})
            .astype({"statefp": int, "countyfp": int})
            .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
            .assign(
                fips=lambda df: (
                    df["statefp"].astype(str) + df["countyfp"].astype(str).str.zfill(3)
                ).astype(int)
            )
            .to_crs(4326)
        )

    def get_geo_filter(self):
        self.get_state_gdf()
        self.geo_filter = self.state_gdf.state_code.unique().tolist() + ["OF"]

    def write_tif_output(self, in_ds, dst_path) -> None:
        """Write a raster to a tif file from a xarray object

        Args:
            in_ds (xr.Dataset): Input xarray dataset containing the raster data.
            dst_path (Path): Path to the output tif file.
            resolution (float, optional): Resolution of the output raster. Defaults to 0.1.

        Returns:
            None

        NOTE: this is being used because the rioxarray.to_raster() function is broken. It
        itroduces noise into the array that I can't explain.

        """
        out_profile = self.gepa_profile.profile
        out_data = in_ds.values
        out_data = np.flip(in_ds.values, axis=1)
        out_profile.update(count=out_data.shape[0])
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(out_data)


class EmiProxyGridder(BaseGridder):

    def __init__(self, emi_proxy_in_data):
        BaseGridder.__init__(self)
        self.gch4i_name = emi_proxy_in_data.gch4i_name
        self.file_type = emi_proxy_in_data.file_type
        self.emi_time_step = emi_proxy_in_data.emi_time_step
        self.emi_geo_level = emi_proxy_in_data.emi_geo_level
        self.emi_id = emi_proxy_in_data.emi_id
        self.proxy_id = emi_proxy_in_data.proxy_id
        self.proxy_time_step = emi_proxy_in_data.proxy_time_step
        self.proxy_time_step = emi_proxy_in_data.proxy_time_step
        self.proxy_has_year_col = emi_proxy_in_data.proxy_has_year_col
        self.proxy_has_month_col = emi_proxy_in_data.proxy_has_month_col
        self.proxy_has_year_month_col = emi_proxy_in_data.proxy_has_year_month_col
        self.proxy_has_rel_emi_col = emi_proxy_in_data.proxy_has_rel_emi_col
        self.proxy_rel_emi_col = emi_proxy_in_data.proxy_rel_emi_col
        self.proxy_geo_level = emi_proxy_in_data.proxy_geo_level
        self.qc_dir = logging_dir / self.gch4i_name
        self.db_path = status_db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.get_status()
        if self.status is None:
            self.status = "not started"
            self.update_status()
        self.base_name = f"{self.gch4i_name}-{self.emi_id}-{self.proxy_id}"
        self.emi_input_path = list(emi_data_dir_path.glob(f"{self.emi_id}.csv"))[0]
        self.proxy_input_path = list(proxy_data_dir_path.glob(f"{self.proxy_id}.*"))[0]
        self.annual_output_path = self.qc_dir / f"{self.base_name}.tif"
        self.has_monthly = (
            self.emi_time_step == "monthly" or self.proxy_time_step == "monthly"
        )
        if self.has_monthly:
            self.monthly_output_path = self.qc_dir / f"{self.base_name}_monthly.tif"
        else:
            self.monthly_output_path = None
        self.time_col = self.get_time_col()
        self.yearly_raster_qc_pass = False
        self.yearly_raster_qc_df = None
        self.monthly_raster_qc_pass = False
        self.monthly_raster_qc_df = None
        logging.info("=" * 83)
        logging.info(f"Gridding {self.base_name}.")
        logging.info(
            f"{self.emi_id} is at {self.emi_geo_level}/{self.emi_time_step} level."
        )
        logging.info(
            f"{self.proxy_id} is at {self.proxy_geo_level}/{self.proxy_time_step} level."
        )

    def get_status(self):
        self.cursor.execute(
            """
            SELECT status FROM gridding_status
            WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?
            """,
            (self.gch4i_name, self.emi_id, self.proxy_id),
        )
        result = self.cursor.fetchone()
        result = result[0] if result else None
        self.status = result

    def get_rel_emi_col(self):
        # if the proxy doesn't have relative emissions, we normalize by the
        # timestep of gridding (year, or year_month)
        if not self.proxy_has_rel_emi_col:
            logging.info(f"{self.proxy_id} adding a relative emissions column.")
            self.proxy_gdf["rel_emi"] = (
                self.proxy_gdf.assign(emis_kt=1)
                .groupby(self.match_cols)["emis_kt"]
                .transform(normalize)
            )
            self.proxy_rel_emi_col = "rel_emi"

    def get_time_col(self):
        if self.emi_time_step == "monthly" or self.proxy_time_step == "monthly":
            self.time_col = "year_month"
        elif self.emi_time_step == "annual" and self.proxy_time_step == "annual":
            self.time_col = "year"

    def get_emi_time_col(self):
        match self.emi_time_step:
            case "monthly":
                self.emi_time_cols = ["year", "month"]
            case "annual":
                self.emi_time_cols = ["year"]
            case _:
                logging.critical(f"emi_time_step {self.emi_time_step} not recognized")

    def get_geo_col(self):
        if self.file_type == "netcdf":
            self.geo_col = "fips"
        elif self.emi_geo_level == "national":
            self.geo_col = None
        elif self.emi_geo_level == "state":
            self.geo_col = "state_code"
        elif self.emi_geo_level == "county":
            self.geo_col = "fips"

    def get_match_cols(self):
        self.get_geo_col()
        self.get_time_col()
        self.match_cols = [self.time_col, self.geo_col]
        self.match_cols = [x for x in self.match_cols if x is not None]

    def read_emi_file(self):
        self.get_emi_time_col()
        match self.emi_geo_level:
            case "national":
                self.emi_cols = self.emi_time_cols + ["ghgi_ch4_kt"]
                self.emi_df = pd.read_csv(self.emi_input_path, usecols=self.emi_cols)
            case "state":
                self.emi_cols = self.emi_time_cols + ["state_code", "ghgi_ch4_kt"]
                self.emi_df = pd.read_csv(
                    self.emi_input_path, usecols=self.emi_cols
                ).query("(state_code.isin(@self.geo_filter)) & (ghgi_ch4_kt > 0)")
            case "county":
                self.emi_cols = self.emi_time_cols + [
                    "state_code",
                    "fips",
                    "ghgi_ch4_kt",
                ]
                self.emi_df = pd.read_csv(
                    self.emi_input_path,
                    usecols=self.emi_cols,
                ).query("(state_code.isin(@self.geo_filter)) & (ghgi_ch4_kt > 0)")
            case _:
                logging.critical(f"emi_geo_level {self.emi_geo_level} not recognized")
        if self.emi_time_step == "monthly":
            self.emi_df = self.emi_df.assign(
                month=lambda df: pd.to_datetime(df["month"], format="%B").dt.month,
                year_month=lambda df: pd.to_datetime(
                    df[["year", "month"]].assign(DAY=1)
                ).dt.strftime("%Y-%m"),
            )
        if self.emi_df.year.unique().shape[0] != len(years):
            self.missing_years = list(set(years) - set(self.emi_df.year.unique()))
            if self.missing_years:
                logging.warning(f"{self.emi_id} is missing years: {self.missing_years}")
                logging.warning(f"these years will be filled with 0s.")
        else:
            self.missing_years = False
        self.get_match_cols()

    def check_vector_proxy_time_geo(self, time_col):

        # check if all the proxy state / time columns are in the emissions data
        # NOTE: this check happens again later, but at a siginificant time cost
        # if the data are large. It is here to catch the error early and break
        # the loop with critical message.

        if self.geo_col:
            grouping_cols = [self.geo_col, time_col]
        else:
            grouping_cols = [time_col]

        proxy_unique = (
            self.proxy_gdf[grouping_cols]
            .drop_duplicates()
            .assign(has_proxy=True)
            .set_index(grouping_cols)
        )
        self.time_geo_qc_df = (
            self.emi_df[grouping_cols]
            .set_index(grouping_cols)
            .join(proxy_unique)
            .fillna({"has_proxy": False})
            .sort_values("has_proxy")
            .reset_index()
        )

        self.time_geo_qc_df.to_csv(
            self.qc_dir / f"{self.base_name}_qc_state_{time_col}.csv"
        )

        if not self.time_geo_qc_df.has_proxy.all():
            if self.geo_col:
                failed_states = (
                    self.time_geo_qc_df[self.time_geo_qc_df["has_proxy"] == False]
                    .groupby(self.geo_col)
                    .size()
                )
                logging.critical(
                    f"QC FAILED: {self.emi_id}, {self.proxy_id}\n"
                    f"proxy state/{time_col} columns do not match emissions\n"
                    "missing states and counts of missing times:\n"
                    # f"{missing_states}\n"
                    f"{failed_states.to_string().replace("\n", "\n\t")}"
                    "\n"
                )
            else:
                logging.critical(
                    f"QC FAILED: {self.emi_id}, {self.proxy_id}\n"
                    "proxy time column does not match emissions\n"
                    "missing times:\n"
                    f"{self.time_geo_qc_df[self.time_geo_qc_df['has_proxy'] == False][time_col].unique()}"
                )
            self.time_geo_qc_pass = False
        else:
            self.time_geo_qc_pass = True

        if self.time_geo_qc_pass:
            logging.info(f"QC PASS: state/{time_col} QC.")
        else:
            logging.critical(f"QC FAIL: state/{time_col} QC.\n")
            self.status = f"failed state/{time_col} QC"
            self.update_status()
            raise ValueError(f"{self.base_name} {self.status}")

    def read_proxy_file(self):
        try:
            self.proxy_gdf = gpd.read_parquet(self.proxy_input_path).reset_index()
        except Exception as e:
            self.status = "error reading proxy"
            self.update_status()
            logging.critical(f"Error reading {self.proxy_id}: {e}\n")
            raise ValueError(f"{self.base_name} {self.status}")

        if self.proxy_gdf.is_empty.any():
            self.status = "proxy has empty geometries"
            self.update_status()
            logging.critical(f"{self.proxy_id} has empty geometries.\n")
            raise ValueError(f"{self.base_name} {self.status}")

        if not self.proxy_gdf.is_valid.all():
            self.status = "proxy has invalid geometries"
            self.update_status()
            logging.critical(f"{self.proxy_id} has invalid geometries.\n")
            raise ValueError(f"{self.base_name} {self.status}")

        # minor formatting issue that some year_months in proxy data were written
        # with underscores instead of dashes.
        if self.proxy_has_year_month_col:
            self.proxy_gdf["year_month"] = pd.to_datetime(
                self.proxy_gdf.year_month.str.replace("_", "-")
            ).dt.strftime("%Y-%m")

        # if the proxy doesn't have a year column, we explode the data out to
        # repeat the same data for every year.
        if not self.proxy_has_year_col:
            logging.info(f"{self.proxy_id} adding a year column.")
            # duplicate the data for all years in years_list
            self.proxy_gdf = self.proxy_gdf.assign(
                year=lambda df: [years for _ in range(df.shape[0])]
            ).explode("year")
        else:
            try:
                self.proxy_gdf = self.proxy_gdf.astype({"year": int})
            except:
                logging.critical(f"{self.proxy_id} year column has NAs.\n")
                self.status = "proxy year has NAs"
                self.update_status()
                raise ValueError(f"{self.base_name} {self.status}")

        # if the proxy data are monthly, but don't have the year_month column,
        # create it.
        if (self.proxy_time_step == "monthly") & (not self.proxy_has_year_month_col):
            logging.info(f"{self.proxy_id} adding a year_month column.")
            # add a year_month column to the proxy data
            try:
                self.proxy_gdf = self.proxy_gdf.assign(
                    year_month=lambda df: pd.to_datetime(
                        df[["year", "month"]].assign(DAY=1)
                    ).dt.strftime("%Y-%m"),
                )
            except ValueError:
                self.proxy_gdf = self.proxy_gdf.assign(
                    month=lambda df: pd.to_datetime(df["month"], format="%b").dt.month,
                    year_month=lambda df: pd.to_datetime(
                        df[["year", "month"]].assign(DAY=1)
                    ).dt.strftime("%Y-%m"),
                )

        # if the proxy data are monthly, but don't have the month column,
        # create it.
        if (self.proxy_time_step == "monthly") & (not self.proxy_has_month_col):
            logging.info(f"{self.proxy_id} adding a month column.")
            # add a month column to the proxy data
            self.proxy_gdf = self.proxy_gdf.assign(
                month=lambda df: pd.to_datetime(df["year_month"]).dt.month
            )

        # TODO: future improvement is to check that all points fall within the state
        # or gridded region, but the operation itself is very time consuming depending
        # on the number of records in a proxy dataset
        # self.proxy_gdf.query(f"state_code.isin({self.geo_filter})").intersects(
        #     self.state_gdf.boundary.union_all()
        # ).all()

    def scale_emi_to_month(self):
        """
        Function to scale the emissions data to a monthly basis
        Parameters:

        - proxy_gdf: GeoDataFrame containing proxy data with relative emissions.
        - emi_df: DataFrame containing emissions data with monthly values.
        Returns:
        - month_check: DataFrame containing the check for monthly emissions.
        """
        # calculate the relative MONTHLY proxy emissions
        logging.info("Calculating monthly scaling factors for emissions data")

        if self.geo_col is not None:
            annual_norm = [self.geo_col, "year"]
        else:
            annual_norm = ["year"]

        if self.file_type == "parquet":
            self.monthly_scaling = (
                self.proxy_gdf.groupby(self.match_cols)["annual_rel_emi"]
                .sum()
                .rename("month_scale")
                .reset_index()
                .assign(
                    year=lambda df: df["year_month"].str.split("-").str[0],
                    month_normed=lambda df: df.groupby(annual_norm)[
                        "month_scale"
                    ].transform(normalize),
                )
                .drop(columns=["month_scale", "year"])
                .set_index(self.match_cols)
            )

        elif self.file_type == "netcdf":
            self.monthly_scaling = (
                self.proxy_ds["annual_rel_emi"]
                .groupby(self.match_cols)
                .sum(dim=...)
                .rename("month_scale")
                .to_dataframe()
                .reset_index()
                .assign(
                    year=lambda df: df["year_month"].str.split("-").str[0],
                    month_normed=lambda df: df.groupby(annual_norm)[
                        "month_scale"
                    ].transform(normalize),
                )
                .drop(columns=["month_scale", "year"])
                .set_index(self.match_cols)
            )
        # display(self.monthly_scaling)
        check_scaling = (
            self.monthly_scaling.reset_index()
            .assign(year=lambda df: df["year_month"].str.split("-").str[0])
            .groupby(annual_norm)["month_normed"]
            .sum()
            .to_frame()
            .query("month_normed > 0")
            .assign(isclose=lambda df: np.isclose(df["month_normed"], 1))
        )
        print("check scaling passed: ", check_scaling["isclose"].all())

        tmp_df = (
            self.emi_df.sort_values(annual_norm)
            .assign(month=lambda df: [list(range(1, 13)) for _ in range(df.shape[0])])
            .explode("month")
            .reset_index(drop=True)
            .assign(
                year_month=lambda df: pd.to_datetime(
                    df[["year", "month"]].assign(DAY=1)
                ).dt.strftime("%Y-%m"),
            )
            .set_index(self.match_cols)
            .join(
                self.monthly_scaling,
                how="outer",
            )
            .assign(
                ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * df["month_normed"],
            )
            .drop(columns=["month_normed"])
            .reset_index()
            .dropna(subset=["ghgi_ch4_kt"])
        )
        tmp_df

        month_check = (
            tmp_df.groupby(annual_norm)["ghgi_ch4_kt"]
            .sum()
            .rename("month_check")
            .to_frame()
            .join(self.emi_df.set_index(annual_norm))
            .assign(isclose=lambda df: np.isclose(df["month_check"], df["ghgi_ch4_kt"]))
        )
        month_check["isclose"].all()

        if not month_check["isclose"].all():
            logging.critical("Monthly emissions do not sum to the expected values")
            raise ValueError(
                "Monthly emissions do not sum to the expected values. Check the log for "
                "details."
            )
        else:
            logging.info("QC PASS: Monthly emissions check!")
            self.emi_df = tmp_df

    def allocate_emissions_to_proxy(self):
        """
        Allocation state emissions by year to all proxies within the state by year.
        NOTE: 2024-06-21: tested with ferro, composting, and aban coal

        Inputs:
            proxy_gdf:
                -   GeoDataFrame: vector proxy data with or without fractional emissions to
                    be used in allocation from state inventory data
            emi_df:
                -   The EPA state level emissions per year, typically read in from
                    the IndDB sheet in the excel workbook
            proxy_has_year:
                -   If the proxy data have a yearly proportional value to use
            use_proportional:
                -   Indicate if the proxy has fractional emissions to be used in the
                    allocation of inventory emissions to the point. For each state /
                    year, the fractional emissions of all points within the state /
                    year are used to allocation inventory emissions.
            proportional_col_name:
                -   the name of the column with proportional emissions.
        Returns:
            -   GeoDataFrame with new column "allocated_ch4_kt" added to proxy_gdf

        """

        self.allocation_gdf = (
            self.proxy_gdf[self.match_cols + ["geometry", self.proxy_rel_emi_col]]
            .merge(self.emi_df, on=self.match_cols, how="right")
            .assign(
                allocated_ch4_kt=lambda df: df[self.proxy_rel_emi_col]
                * df["ghgi_ch4_kt"]
            )
        )

    def QC_proxy_allocation(self, plot=False, plot_path=None):
        """take proxy emi allocations and check against state inventory"""

        emi_sums = (
            self.emi_df.groupby(self.match_cols)["ghgi_ch4_kt"].sum().reset_index()
        )

        relative_tolerance = 0.0001
        self.allocation_qc_df = (
            self.allocation_gdf.groupby(self.match_cols)["allocated_ch4_kt"]
            .sum()
            .reset_index()
            .merge(emi_sums, on=self.match_cols, how="outer")
            .assign(
                isclose=lambda df: np.isclose(
                    df["allocated_ch4_kt"], df["ghgi_ch4_kt"]
                ),
                diff=lambda df: np.abs(df["allocated_ch4_kt"] - df["ghgi_ch4_kt"])
                / ((df["allocated_ch4_kt"] + df["ghgi_ch4_kt"]) / 2),
                qc_pass=lambda df: (df["diff"] < relative_tolerance),
            )
        )

        # in the cases where the difference is NaN (where the inventory emission value is 0
        # and the allocated value is 0), we fall back on numpy isclose to define
        # if the value is close enough to pass
        self.allocation_qc_df["qc_pass"] = self.allocation_qc_df["isclose"].where(
            self.allocation_qc_df["diff"].isna(), self.allocation_qc_df["qc_pass"]
        )

        self.allocation_qc_df.to_csv(
            self.qc_dir / f"{self.base_name}_allocation_qc.csv"
        )

        self.allocation_qc_pass = self.allocation_qc_df.qc_pass.all()

        if self.allocation_qc_pass:
            logging.info("QC PASS: all proxy emission by state/year equal (isclose)")
        else:
            logging.critical(
                f"QC FAIL: {self.emi_id}, {self.proxy_id}. allocation failed."
            )
            logging.info("states and years with emissions that don't match")
            if self.geo_col:
                unique_state_codes = self.emi_df[
                    ~self.emi_df["state_code"].isin(self.proxy_gdf["state_code"])
                ]["state_code"].unique()
                logging.warning(
                    f"states with no proxy points in them: {unique_state_codes}"
                )
                logging.info(
                    (
                        "states with unaccounted emissions: "
                        f"{self.allocation_qc_df[~self.allocation_qc_df['isclose']]['state_code'].unique()}"
                    )
                )
            else:
                logging.critical(
                    f"QC FAIL: {self.emi_id}, {self.proxy_id}. annual allocation failed."
                )
                self.status = "failed allocation QC"
                self.update_status()
                raise ValueError(f"{self.base_name} {self.status}")

        if plot and self.allocation_qc_pass and self.geo_col and self.time_col:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            fig.suptitle("compare inventory to allocated emissions by state")
            sns.lineplot(
                data=self.allocation_qc_df,
                x=self.time_col,
                y="allocated_ch4_kt",
                hue=self.geo_col,
                palette="tab20",
                legend=False,
                ax=axs[0],
            )
            axs[0].set(title="allocated emissions")

            sns.lineplot(
                data=self.allocation_qc_df,
                x=self.time_col,
                y="ghgi_ch4_kt",
                hue=self.geo_col,
                palette="tab20",
                legend=False,
                ax=axs[1],
            )
            axs[1].set(title="inventory emissions")
            plt.savefig(plot_path, dpi=300)
            plt.close()
        elif plot and self.allocation_qc_pass and self.time_col and not self.geo_col:
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            fig.suptitle("compare inventory to allocated emissions by state")
            sns.lineplot(
                data=self.allocation_qc_df,
                x=self.time_col,
                y="allocated_ch4_kt",
                legend=False,
                ax=axs[0],
            )
            axs[0].set(title="allocated emissions")

            sns.lineplot(
                data=self.allocation_qc_df,
                x=self.time_col,
                y="ghgi_ch4_kt",
                legend=False,
                ax=axs[1],
            )
            axs[1].set(title="inventory emissions")
            plt.savefig(plot_path, dpi=300)
            plt.close()

    def create_empty_proxy(self):
        proxy_times = self.proxy_gdf[self.time_col].unique()
        empty_shape = (
            len(proxy_times),
            self.gepa_profile.height,
            self.gepa_profile.width,
        )
        empty_data = np.zeros(empty_shape)
        if self.time_col == "year_month":
            self.proxy_ds = xr.DataArray(
                empty_data,
                dims=[self.time_col, "y", "x"],
                coords={
                    self.time_col: proxy_times,
                    "y": self.gepa_profile.y,
                    "x": self.gepa_profile.x,
                },
                name="results",
            ).to_dataset(name="results")
            self.proxy_ds = self.proxy_ds.assign_coords(
                year=(
                    "year_month",
                    pd.to_datetime(self.proxy_ds.year_month.values).year,
                ),
                month=(
                    "year_month",
                    pd.to_datetime(self.proxy_ds.year_month.values).month,
                ),
            )
        elif self.time_col == "year":
            self.proxy_ds = xr.DataArray(
                empty_data,
                dims=[self.time_col, "y", "x"],
                coords={
                    self.time_col: proxy_times,
                    "y": self.gepa_profile.y,
                    "x": self.gepa_profile.x,
                },
                name="results",
            ).to_dataset(name="results")

    def grid_vector_data(self):
        if self.allocation_gdf.empty:
            logging.warning(
                f"{self.proxy_id} allocation is empty. likely no"
                f"emissions data for {self.emi_id}"
            )
            self.create_empty_proxy()
        else:
            try:
                # STEP X: GRID EMISSIONS
                # turn the vector proxy into a grid
                self.grid_allocated_emissions()
            except Exception as e:
                logging.critical(
                    f"{self.emi_id}, {self.proxy_id} gridding failed {e}\n"
                )
                self.status = "gridding failed"
                self.update_status()
                raise ValueError(f"{self.base_name} {self.status}")

    def update_status(self):
        self.cursor.execute(
            """
        INSERT INTO gridding_status (gch4i_name, emi_id, proxy_id, status)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(gch4i_name, emi_id, proxy_id) DO UPDATE SET status=excluded.status
        """,
            (self.gch4i_name, self.emi_id, self.proxy_id, self.status),
        )
        self.conn.commit()

    def grid_allocated_emissions(self):
        # if the proxy data do not nest nicely into rasters cells, we have to do some
        # disaggregation of the vectors to get the sums to equal during rasterization
        # so we need the cell geodataframe to do that. Only load the cell_gdf if we need
        # to do disaggregation. project to equal are for are calculations
        # if not (proxy_gdf.geometry.type == "Points").all():
        cell_gdf = get_cell_gdf().to_crs("ESRI:102003")

        # create a dictionary to hold the rasterized emissions
        ch4_kt_result_rasters = {}
        for time_var, time_data in self.allocation_gdf.groupby(self.time_col):
            # print(time_var)

            # orig_emi_val = data["allocated_ch4_kt"].sum()
            # if we need to disaggregate non-point data. Maybe extra, but this accounts for
            # potential edge cases where the data change from year to year.

            if not (time_data.geometry.type == "Points").all():
                data_to_concat = []
                for geom_type, geom_data in time_data.groupby(time_data.geom_type):
                    # print(geom_type)
                    # regular points can just be passed on "as is"
                    if geom_type == "Point":
                        # get the point data
                        # print("doing point data")
                        point_data = geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                        # print(point_data.is_empty.any())
                        data_to_concat.append(point_data)
                    # if we have multipoint data, we need to disaggregate the emissions
                    # across the individual points and align them with the cells
                    elif geom_type == "MultiPoint":
                        # print("doing multipoint data")
                        # ref_val = geom_data["allocated_ch4_kt"].sum()
                        multi_point_data = (
                            geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                            .to_crs("ESRI:102003")
                            .explode(index_parts=True)
                            .reset_index()
                            .assign(
                                allocated_ch4_kt=lambda df: df.groupby("level_0")[
                                    "allocated_ch4_kt"
                                ].transform(lambda x: x / len(x)),
                            )
                            .overlay(cell_gdf)
                            .reset_index()
                            .assign(
                                geometry=lambda df: df.centroid,
                            )
                            .groupby("index")
                            .agg({"allocated_ch4_kt": "sum", "geometry": "first"})
                        )
                        multi_point_data = gpd.GeoDataFrame(
                            multi_point_data, geometry="geometry", crs="ESRI:102003"
                        ).to_crs(4326)
                        # print(ref_val)
                        # print(multi_point_data["allocated_ch4_kt"].sum())
                        # print(f" any empty data: {multi_point_data.is_empty.any()}")
                        data_to_concat.append(multi_point_data)
                    # if the data are any kind of lines, compute the relative length within
                    # each cell
                    elif "Line" in geom_type:
                        # print("doing line data")
                        line_data = (
                            geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                            # project to equal area for area calculations
                            .to_crs("ESRI:102003")
                            # calculate the original proxy length
                            .assign(orig_len=lambda df: df.length)
                            # overlay the proxy with the cells, this results in splitting the
                            # original proxies across any intersecting cells
                            .overlay(cell_gdf)
                            # # calculate the now partial proxy length, then divide the partial
                            # # proxy by the original proxy and multiply by the original
                            # # allocated emissions to the get the partial/disaggregated new emis.
                            .assign(
                                len=lambda df: df.length,
                                allocated_ch4_kt=lambda df: (df["len"] / df["orig_len"])
                                * df["allocated_ch4_kt"],
                                #     # make this new shape a point, that fits nicely inside the cell
                                #     # and we don't have to worry about boundary/edge issues with
                                #     # polygon / cell alignment
                                geometry=lambda df: df.centroid,
                            )
                            # back to 4326 for rasterization
                            .to_crs(4326).loc[:, ["geometry", "allocated_ch4_kt"]]
                        )
                        # print(f" any empty data: {line_data.is_empty.any()}")
                        data_to_concat.append(line_data)
                    # if the data are any kind of  polygons, compute the relative area in
                    # each cell
                    elif "Polygon" in geom_type:
                        # print("doing polygon data")
                        polygon_data = (
                            geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                            # project to equal area for area calculations
                            .to_crs("ESRI:102003")
                            # calculate the original proxy area
                            .assign(orig_area=lambda df: df.area)
                            # overlay the proxy with the cells, this results in splitting the
                            # original proxies across any intersecting cells
                            .overlay(cell_gdf)
                            # calculate the now partial proxy area, then divide the partial
                            # proxy by the original proxy and multiply by the original
                            # allocated emissions to the get the partial/disaggregated new emis.
                            .assign(
                                area=lambda df: df.area,
                                allocated_ch4_kt=lambda df: (
                                    df["area"] / df["orig_area"]
                                )
                                * df["allocated_ch4_kt"],
                                # make this new shape a point, that fits nicely inside the cell
                                # and we don't have to worry about boundary/edge issues with
                                # polygon / cell alignment
                                geometry=lambda df: df.centroid,
                            )
                            # back to 4326 for rasterization
                            .to_crs(4326).loc[:, ["geometry", "allocated_ch4_kt"]]
                        )
                        # print(polygon_data["allocated_ch4_kt"].sum())
                        # print(f" any empty data: {polygon_data.is_empty.any()}")
                        data_to_concat.append(polygon_data)
                    else:
                        raise ValueError(
                            "I don't support that geometry type. Need to fix"
                        )
                # concat the data back together
                ready_data = pd.concat(data_to_concat)
            else:
                ready_data = time_data.loc[:, ["geometry", "allocated_ch4_kt"]]

            # Check for empty or invalid geometries
            invalid_geometries = ready_data[
                ready_data.geometry.is_empty | ~ready_data.geometry.is_valid
            ]

            if not invalid_geometries.empty:
                logging.warning(
                    f"Found {invalid_geometries.shape[0]} invalid or empty geometries with a total allocated_ch4_kt of "
                    f"{invalid_geometries['allocated_ch4_kt'].sum()}"
                )
                # Drop invalid or empty geometries
                ready_data = ready_data[
                    ~(ready_data.geometry.is_empty | ~ready_data.geometry.is_valid)
                ]

            # print("compare pre and post processing sums:")
            # print("pre data:  ", time_data[["allocated_ch4_kt"]].sum())
            # print("post data: ", ready_data[["allocated_ch4_kt"]].sum())

            # now rasterize the emissions and sum within cells
            ch4_kt_raster = rasterize(
                shapes=[
                    (shape, value)
                    for shape, value in ready_data[
                        ["geometry", "allocated_ch4_kt"]
                    ].values
                ],
                out_shape=self.gepa_profile.arr_shape,
                fill=0,
                transform=self.gepa_profile.profile["transform"],
                dtype=np.float64,
                merge_alg=rasterio.enums.MergeAlg.add,
            )
            ch4_kt_result_rasters[time_var] = ch4_kt_raster

        time_var = list(ch4_kt_result_rasters.keys())
        arr_stack = np.stack(list(ch4_kt_result_rasters.values()), axis=0)
        arr_stack = np.flip(arr_stack, axis=1)
        self.proxy_ds = xr.Dataset(
            {"results": ([self.time_col, "y", "x"], arr_stack)},
            coords={
                self.time_col: time_var,
                "y": self.gepa_profile.y,
                "x": self.gepa_profile.x,
            },
        )
        if self.time_col == "year_month":
            self.proxy_ds = self.proxy_ds.assign_coords(
                year=(self.time_col, [int(x.split("-")[0]) for x in time_var]),
                month=(self.time_col, [int(x.split("-")[1]) for x in time_var]),
            )

    def prepare_vector_proxy(self):
        self.read_proxy_file()
        self.get_rel_emi_col()
        # check that the proxy and emi files have matching state years
        # NOTE: this is going to arise when the proxy data are lacking adequate
        # spatial or temporal coverage. Failure here will require finding new data
        # and/or filling in missing state/years.
        if (self.proxy_time_step == "monthly") & (self.emi_time_step == "annual"):
            self.check_vector_proxy_time_geo("year")
            self.scale_emi_to_month()
            self.check_vector_proxy_time_geo("year_month")
        elif self.proxy_time_step == self.emi_time_step:
            self.check_vector_proxy_time_geo(self.time_col)
        self.allocate_emissions_to_proxy()
        self.QC_proxy_allocation()
        self.grid_vector_data()

    def make_emi_grid(self):
        """
        Function to create a 3D array of emissions data for each year and month
        Parameters:
        - emi_df: DataFrame containing emissions data with yearly or monthly values.
        - admin_gdf: GeoDataFrame containing administrative boundaries.
        Returns:
        - emi_rasters_3d: 3D array of emissions data for each year or year + month.
        """
        # to calculate emissions for a proxy array, we make the emissions into an
        # array of the same shape as the proxy array, and then multiply the two

        transform = self.gepa_profile.profile["transform"]
        out_shape = self.gepa_profile.arr_shape
        time_total = self.emi_df[self.time_col].unique().shape[0]
        # this turns each year of the emissions data into a raster of the same shape
        # as the proxy in x and y. We have already aligned the time steps so the
        # number of emissions rasters should match the number of proxy rasters in
        # that dimension.
        # NOTE: this is kind of slow.

        def emi_rasterize(e_df, adm_gdf, out_shape, transform, year):
            emi_gdf = e_df.merge(adm_gdf, on="fips")
            emi_raster = rasterize(
                [
                    (geom, value)
                    for geom, value in zip(emi_gdf.geometry, emi_gdf.ghgi_ch4_kt)
                ],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype="float32",
                # merge_alg=rasterio.enums.MergeAlg.add,
            )
            return year, emi_raster

        parallel = Parallel(n_jobs=-1, return_as="generator")
        emi_gen = parallel(
            delayed(emi_rasterize)(
                emi_time_df, self.admin_gdf, out_shape, transform, year
            )
            for year, emi_time_df in self.emi_df.groupby(self.time_col)
        )

        emi_rasters = []
        emi_years = []
        for year, raster in tqdm(
            emi_gen,
            total=time_total,
            desc="making emi grids",
        ):
            emi_rasters.append(raster)
            emi_years.append(year)

        emi_rasters_3d = np.stack(emi_rasters, axis=0)
        # flip the y axis as xarray expects
        emi_rasters_3d = np.flip(emi_rasters_3d, axis=1)

        self.emi_xr = xr.DataArray(
            emi_rasters_3d,
            dims=[self.time_col, "y", "x"],
            coords={
                self.time_col: emi_years,
                "y": self.gepa_profile.y,
                "x": self.gepa_profile.x,
            },
        )

    def check_raster_proxy_time_geo(self):
        # check that the proxy and emi files have matching state years
        # NOTE: this is going to arise when the proxy data are lacking adequate
        # spatial or temporal coverage. Failure here will require finding new data
        # and/or filling in missing state/years.

        if self.geo_col is not None:
            grouper_cols = [self.geo_col, "year"]
        else:
            grouper_cols = [self.time_col]

        # if self.time_col == "year_month":
        #     self.proxy_ds = self.proxy_ds.assign_coords(
        #         year_month=pd.to_datetime(
        #             pd.DataFrame(
        #                 {
        #                     "year": self.proxy_ds.year.values,
        #                     "month": self.proxy_ds.month.values,
        #                     "day": 1,
        #                 }
        #             )
        #         ).dt.strftime("%Y-%m")
        #     )

        rel_emi_check = (
            self.proxy_ds["rel_emi"]
            .groupby(grouper_cols)
            .sum()
            .to_dataframe()
            .reset_index()
            .set_index(grouper_cols)
            .assign(has_proxy=lambda df: df["rel_emi"] > 0)
        )
        final_df = self.emi_df.set_index(grouper_cols).join(rel_emi_check)
        self.time_geo_qc_df = final_df[
            (final_df.ghgi_ch4_kt > 0) & (final_df.rel_emi < 0.9)
        ]
        self.time_geo_qc_df.to_csv(self.qc_dir / f"{self.base_name}_qc_state_year.csv")
        if len(final_df.query("(ghgi_ch4_kt > 0) & ~has_proxy")) > 0:
            if self.geo_col:
                missing_states = self.time_geo_qc_df.state_code.value_counts()
                logging.critical(
                    f"QC FAILED: {self.emi_id}, {self.proxy_id}\n"
                    "proxy state/year columns do not match emissions\n"
                    "missing states and counts of missing times:\n"
                    f"{missing_states.to_string().replace("\n", "\n\t")}"
                    "\n"
                )
            else:
                logging.critical(
                    f"QC FAILED: {self.emi_id}, {self.proxy_id}\n"
                    "proxy time column does not match emissions\n"
                    "missing times:\n"
                    f"{self.time_geo_qc_df[self.time_geo_qc_df['has_proxy'] == False][self.time_col].unique()}"
                )
            self.status = "failed state/year QC"
            self.update_status()
            self.time_geo_qc_pass = False
            raise ValueError(f"{self.base_name} {self.status}")
        else:
            self.time_geo_qc_pass = True

    def prepare_raster_proxy(self):
        # read the proxy file
        self.proxy_ds = xr.open_dataset(
            self.proxy_input_path
        )  # .rename({"geoid": geo_col})
        self.proxy_ds = self.proxy_ds.assign_coords(
            x=("x", self.gepa_profile.x), y=("y", self.gepa_profile.y)
        )

        if "fips" not in self.emi_df.columns:
            self.emi_df = self.emi_df.merge(
                self.state_gdf[["state_code", "fips"]], on="state_code", how="left"
            )
        if "fips" not in self.proxy_ds.coords:
            if "geoid" in self.proxy_ds.coords:
                self.proxy_ds = self.proxy_ds.rename({"geoid": "fips"})
            elif "statefp" in self.proxy_ds.coords:
                self.proxy_ds = self.proxy_ds.rename({"statefp": "fips"})

        # if the emi is month and the proxy is annual, we expand the dimensions of
        # the proxy, repeating the year values for every month in the year
        # we stack the year/month dimensions into a single year_month so that
        # it aligns with the emissions data as a time x X x Y array.
        if self.emi_time_step == "monthly" and self.proxy_time_step == "annual":
            self.proxy_ds = (
                self.proxy_ds.expand_dims(dim={"month": np.arange(1, 13)}, axis=0)
                .stack({"year_month": ["year", "month"]}, create_index=False)
                .sortby(["year_month", "y", "x"])
            )
            year_months = pd.to_datetime(
                self.proxy_ds[["year", "month"]].to_dataframe().assign(day=1)
            ).dt.strftime("%Y-%m")
            self.proxy_ds = self.proxy_ds.assign_coords(
                year_month=("year_month", year_months)
            ).transpose("year_month", "y", "x")

        elif (self.proxy_time_step == "monthly") & (self.emi_time_step == "annual"):
            print("DEBUG: scaling emis")
            self.scale_emi_to_month()

        # check that the proxy and emi files have matching state years
        # NOTE: this is going to arise when the proxy data are lacking adequate
        # spatial or temporal coverage. Failure here will require finding new data
        # and/or filling in missing state/years.
        self.check_raster_proxy_time_geo()

        match self.emi_geo_level:
            case "state":
                self.get_state_gdf()
                self.admin_gdf = self.state_gdf
            case "county":
                self.get_county_gdf()
                self.admin_gdf = self.county_gdf
        # NOTE: this is slow.
        self.make_emi_grid()

        # here we are going to remove missing years from the proxy dataset
        # so we can stack it against the emissions dataset
        if self.missing_years:
            self.proxy_ds = self.proxy_ds.drop_sel(year=list(self.missing_years))

        # assign the emissions array to the proxy dataset
        # proxy_ds["emissions"] = ([time_col, "y", "x"], emi_xr)
        self.proxy_ds["emissions"] = (self.emi_xr.dims, self.emi_xr.data)

        # # look to make sure the emissions loaded in the correct orientation
        # proxy_ds["emissions"].sel(year_month=(2020, 1)).where(lambda x: x > 0).plot(
        #     cmap="hot"
        # )
        # proxy_ds["emissions"].sel(year=2020).where(lambda x: x > 0).plot(cmap="hot")
        # plt.show()
        # calculate the emissions for the proxy array by multiplying the proxy by
        # the gridded emissions data
        self.proxy_ds["results"] = (
            self.proxy_ds[self.proxy_rel_emi_col] * self.proxy_ds["emissions"]
        )

    def QC_emi_raster_sums(self, QC_time_col):
        """compares yearly array sums to inventory emissions"""

        # emi_sum_check = (
        #     self.emi_df.groupby(self.time_col)["ghgi_ch4_kt"].sum().to_frame()
        # )
        # proxy_sum_check = self.proxy_ds.sum(dim=["x", "y"]).to_dataframe()

        grid_result_df = (
            self.proxy_ds["results"].groupby(QC_time_col).sum(dim=...).to_dataframe()
        )
        emi_result_df = self.emi_df.groupby(QC_time_col)["ghgi_ch4_kt"].sum()

        if QC_time_col == "year_month" and isinstance(
            grid_result_df.index, pd.MultiIndex
        ):
            grid_result_df.index = grid_result_df.index.map(
                lambda x: f"{x[0]}-{x[1]:02d}"
            )
            relative_tolerance = 0.0001
        else:
            relative_tolerance = 0.0001

        # The relative tolerance is the maximum allowable difference between the two values
        # as a fraction of the average of the two values. It is used to determine if
        # the two values are close enough to be considered equal.
        # The default value is 1e-5, which means that the two values must be within 0.01%
        # of each other to be considered equal.
        raster_qc_df = (
            emi_result_df.to_frame()
            .join(grid_result_df)
            .assign(
                diff=lambda x: x["ghgi_ch4_kt"] - x["results"],
                rel_diff=lambda df: np.abs(df["ghgi_ch4_kt"] - df["results"])
                / ((df["ghgi_ch4_kt"] + df["results"]) / 2),
                isclose=lambda df: np.isclose(
                    df["ghgi_ch4_kt"], df["results"], atol=0, rtol=1e-5
                ),
                qc_pass=lambda df: (df["rel_diff"] < relative_tolerance),
            )
        )

        # in the cases where the difference is NaN (where the inventory emission value is 0
        # and the allocated value is 0), we fall back on numpy isclose to define
        # if the value is close enough to pass
        raster_qc_df["qc_pass"] = raster_qc_df["isclose"].where(
            raster_qc_df["rel_diff"].isna(), raster_qc_df["qc_pass"]
        )
        qc_pass = raster_qc_df.qc_pass.all()

        if QC_time_col == "year_month":
            out_path = self.qc_dir / f"{self.base_name}_emi_grid_qc_monthly.csv"
            self.monthly_raster_qc_df = raster_qc_df
            self.monthly_raster_qc_pass = qc_pass
            self.monthly_raster_qc_df.to_csv(out_path)
        else:
            out_path = self.qc_dir / f"{self.base_name}_emi_grid_qc.csv"
            self.yearly_raster_qc_df = raster_qc_df
            self.yearly_raster_qc_pass = qc_pass
            self.yearly_raster_qc_df.to_csv(out_path)

        if qc_pass:
            logging.info(f"QC PASS: all gridded emission by {QC_time_col}.")
        else:
            logging.critical(
                "QC FAIL: gridded emissions do not equal inventory emissions."
            )
            logging.info(
                "\t"
                + raster_qc_df[~raster_qc_df.qc_pass].to_string().replace("\n", "\n\t")
            )

    def fill_missing_year_months(self):
        # Ensure proxy_ds has all year_months in the range 2012-2022
        # in cases where we have expanded the annual emissions to match a monthly proxy,
        # we need to ensure that if the proxy is missing any months, we assume there are
        # no emissions, but we still need a layer representing 0 in that month.
        # this will fill in any missign years
        if self.time_col == "year":
            proxy_year_month = self.proxy_ds["year"].values
            all_year_months = years
        elif self.time_col == "year_month":
            all_year_months = pd.date_range(
                start=f"{min(years)}-01", end=f"{max(years)}-12-31", freq="ME"
            ).strftime("%Y-%m")

            if isinstance(self.proxy_ds.indexes["year_month"], pd.MultiIndex):
                proxy_year_month = self.proxy_ds.indexes["year_month"].map(
                    lambda x: f"{x[0]}-{x[1]:02d}"
                )
            else:
                proxy_year_month = self.proxy_ds["year_month"].values

        missing_year_months = set(all_year_months) - set(proxy_year_month)

        if missing_year_months:
            logging.info(f"Filling missing year_months: {missing_year_months}")
            empty_array = np.zeros_like(
                self.proxy_ds["results"].isel(year_month=0).values
            )
            if self.time_col == "year":
                for year in missing_year_months:
                    if isinstance(self.proxy_ds.indexes["year_month"], pd.MultiIndex):
                        fill_value = (year, 1)
                    else:
                        fill_value = year
                    missing_da = xr.Dataset(
                        {"results": (["y", "x"], empty_array)},
                        coords={
                            "year_month": [fill_value],
                            "year": year,
                            "y": self.proxy_ds["y"],
                            "x": self.proxy_ds["x"],
                        },
                    )
                    self.proxy_ds = xr.concat(
                        [
                            self.proxy_ds,
                            missing_da,
                        ],
                        dim="year_month",
                    )
                self.proxy_ds = self.proxy_ds.sortby("year")
            elif self.time_col == "year_month":
                for year_month in missing_year_months:
                    year, month = map(int, year_month.split("-"))
                    if isinstance(self.proxy_ds.indexes["year_month"], pd.MultiIndex):
                        fill_value = (year, month)
                    else:
                        fill_value = year_month

                    missing_da = xr.Dataset(
                        {"results": (["y", "x"], empty_array)},
                        coords={
                            "year_month": [fill_value],
                            "year": year,
                            "month": month,
                            "y": self.proxy_ds["y"],
                            "x": self.proxy_ds["x"],
                        },
                    )
                    self.proxy_ds = xr.concat(
                        [
                            self.proxy_ds,
                            missing_da,
                        ],
                        dim="year_month",
                    )
                self.proxy_ds = self.proxy_ds.sortby("year_month")

    def qc_and_write_output(self):
        # The expectation at this step is that regardless of a parquet or gridded input
        # we have a proxy_ds that is a DataArray with dimensions of time, y, x
        # and a results variable that is the product of the proxy and the emissions.

        # in some cases, especially when we have a monthly proxy and annaul emissions,
        # we end up with months that do not exist. It is also possible for an emissions
        # source to be all zeros for a year, and also missing. So this function will
        # fill in the missing years and months with zeros.
        self.fill_missing_year_months()
        # if the time step is monthly, we need to also get yearly emissions data. We QC
        # the monthly data.
        if self.has_monthly:
            self.yearly_results = self.proxy_ds["results"].groupby("year").sum()
            self.QC_emi_raster_sums(self.time_col)
            if self.monthly_raster_qc_pass:
                self.write_tif_output(
                    self.proxy_ds["results"], self.monthly_output_path
                )
                logging.info(f"{self.base_name} monthly gridding complete.\n")
            else:
                logging.critical(f"{self.base_name} failed monthly raster QC.\n")
        else:
            # if the time step is annual, we need to get the annual emissions data. Here
            # this just gives it a new name, but it is the same as the results variable.
            self.yearly_results = self.proxy_ds["results"]
        # now we QC the annual data. This is the final QC step.
        self.QC_emi_raster_sums("year")
        if self.yearly_raster_qc_pass:
            self.write_tif_output(self.yearly_results, self.annual_output_path)
            logging.info(f"{self.base_name} annual gridding complete.\n")
        else:
            logging.critical(f"{self.base_name} failed annual raster QC.\n")

        if self.has_monthly:
            if all([self.monthly_raster_qc_pass, self.yearly_raster_qc_pass]):
                self.status = "complete"
                self.update_status()
            elif self.monthly_raster_qc_pass and not self.yearly_raster_qc_pass:
                self.status = "monthly complete, annual failed"
                self.update_status()
            elif not self.monthly_raster_qc_pass and self.yearly_raster_qc_pass:
                self.status = "monthly failed, annual complete"
                self.update_status()
            elif not self.monthly_raster_qc_pass and not self.yearly_raster_qc_pass:
                self.status = "monthly failed, annual failed"
                self.update_status()
        else:
            if self.yearly_raster_qc_pass:
                self.status = "complete"
                self.update_status()

            else:
                self.status = "failed"
                self.update_status()
                raise ValueError(f"{self.base_name} {self.status}")

    def run_gridding(self):
        self.read_emi_file()
        if self.file_type == "parquet":
            self.prepare_vector_proxy()
        elif self.file_type == "netcdf":
            self.prepare_raster_proxy()
        self.qc_and_write_output()
        self.conn.close()


class GroupGridder(BaseGridder):
    # custom_colormap = colors.LinearSegmentedColormap.from_list(
    #     name="custom_colormap",
    #     colors=[
    #         "#2166AC",
    #         "#4393C3",
    #         "#92C5DE",
    #         "#D1E5F0",
    #         "#F7F7F7",
    #         "#FDDBC7",
    #         "#F4A582",
    #         "#D6604D",
    #         "#B2182B",
    #     ],
    #     N=3000,
    # )

    # The EPA color map from their V2 plots
    emi_custom_colormap = colors.LinearSegmentedColormap.from_list(
        name="emi_cmap",
        colors=[
            "#6F4C9B",
            "#6059A9",
            "#5568B8",
            "#4E79C5",
            "#4D8AC6",
            "#4E96BC",
            "#549EB3",
            "#59A5A9",
            "#60AB9E",
            "#69B190",
            "#77B77D",
            "#8CBC68",
            "#A6BE54",
            "#BEBC48",
            "#D1B541",
            "#DDAA3C",
            "#E49C39",
            "#E78C35",
            "#E67932",
            "#E4632D",
            "#DF4828",
            "#DA2222",
            "#B8221E",
            "#95211B",
            "#721E17",
            "#521A13",
        ],
        N=3000,
    )

    def __init__(self, group_name, in_data, dst_dir):
        BaseGridder.__init__(self)
        self.group_name = group_name
        self.data_df = in_data
        self.dst_dir = dst_dir
        self.qc_dir = logging_dir / self.group_name
        self.annual_source_count = self.data_df.shape[0]
        self.get_monthly_source_count()
        if self.monthly_source_count > 0:
            self.has_monthly = True
            self.time_col = "year_month"
            self.monthly_scale_output_path = (
                self.dst_dir / f"monthly_scaling/{self.group_name}_monthly_scaling.tif"
            )
        self.v2_name = self.data_df["v2_key"].iloc[0]
        self.tif_flux_output_path = self.dst_dir / f"{self.group_name}_ch4_emi_flux.tif"
        self.tif_kt_output_path = (
            self.dst_dir / f"{self.group_name}_ch4_kt_per_year.tif"
        )
        self.nc_flux_output_path = self.dst_dir / f"{self.group_name}_ch4_emi_flux.nc"
        self.get_geo_filter()
        self.relative_tolerance = 0.0001

        # IPCC_ID, SOURCE_NAME = g_name.split("_", maxsplit=1)
        # netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
        # netcdf_description = (
        #     f"Gridded EPA Inventory - {g_name} - "
        #     f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
        # )
        # netcdf_title = f"CH4 emissions from {source_name} gridded to 1km x 1km"
        # write_ncdf_output(
        #     ch4_flux_result_da,
        #     nc_flux_output_path,
        #     netcdf_title,
        #     netcdf_description,
        # )

    def get_monthly_source_count(self):
        self.monthly_source_count = (
            self.data_df[["emi_time_step", "proxy_time_step"]]
            .eq("monthly")
            .any(axis=1)
            .sum()
        )

    def get_source_QC_df(self):

        qc_files = []
        for row in self.data_df.itertuples():
            base_name = f"{row.gch4i_name}-{row.emi_id}-{row.proxy_id}"
            result = list(self.qc_dir.glob(f"{base_name}_emi_grid_qc.csv"))
            qc_files.extend(result)

        qc_files = [x for x in qc_files if "monthly" not in x.name]
        if len(qc_files) != self.annual_source_count:
            # raise ValueError(
            #     f"{self.group_name} has {len(qc_files)} qc files, but should have "
            #     f"{self.annual_source_count}."
            # )
            warnings.warn(
                f"{self.group_name} has {len(qc_files)} qc files, but should have "
                f"{self.annual_source_count}."
            )

        self.all_source_qc_df = pd.concat([pd.read_csv(x) for x in qc_files], axis=0)

        # NOTE: this is a minor hack to deal with a column name change in the QC files
        # this should be removed once all the QC files have been updated.
        if ("rel_diff" in self.all_source_qc_df.columns) and (
            "diff" in self.all_source_qc_df.columns
        ):
            self.all_source_qc_df["diff"] = self.all_source_qc_df["rel_diff"]

        self.source_qc_df = (
            self.all_source_qc_df.groupby("year")
            .agg(
                {
                    "ghgi_ch4_kt": "sum",
                    "results": "sum",
                    "isclose": "all",
                    "diff": "sum",
                }
            )
            .rename(
                columns={
                    "ghgi_ch4_kt": "source_emi_sum",
                    "results": "source_grid_sum",
                    "isclose": "source_passed",
                    "diff": "source_diff_sum",
                }
            )
        )

    def get_group_emi_df(self):
        """get all the original emissions files and sum up emissions by year."""
        emi_results_list = []
        for row in self.data_df.itertuples():
            emi_df = pd.read_csv(emi_data_dir_path / f"{row.emi_id}.csv")
            try:
                emi_df = emi_df.query(
                    f"(state_code.isin({self.geo_filter})) & (ghgi_ch4_kt > 0)"
                )
            except:
                print("national emissions")
            emi_results_list.append(emi_df)
        self.all_emi_results_df = pd.concat(emi_results_list, axis=0)
        self.emi_group_year_df = (
            self.all_emi_results_df.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
        )

    def get_input_raster_paths(self):
        """get all the input emi/proxy pair paths."""
        all_raster_list = []
        for row in self.data_df.itertuples():
            base_name = f"{row.gch4i_name}-{row.emi_id}-{row.proxy_id}"
            result = list(self.qc_dir.glob(f"{base_name}*.tif"))
            all_raster_list.extend(result)

        # split the lists into annual and monthly
        annual_raster_list = [
            raster for raster in all_raster_list if "monthly" not in raster.name
        ]

        # check that we got the number of files we expected
        if annual_raster_list:
            if len(annual_raster_list) == self.annual_source_count:
                self.annual_raster_list = annual_raster_list
            else:
                raise ValueError(
                    f"only found {len(annual_raster_list)} annual rasters. "
                    f"Expected {self.annual_source_count}."
                )
        else:
            raise ValueError("No annual rasters found.")

        # if we expect monthly raster files, get the list of paths and check that we
        # have the right number of files
        if self.monthly_source_count > 0:
            monthly_raster_list = [
                raster for raster in all_raster_list if "monthly" in raster.name
            ]
            if monthly_raster_list:
                if len(monthly_raster_list) == self.monthly_source_count:
                    self.monthly_raster_list = monthly_raster_list
                else:
                    raise ValueError(
                        f"only found {len(monthly_raster_list)} annual rasters. "
                        f"Expected {self.monthly_source_count}."
                    )
            else:
                raise ValueError(
                    f"No monthly rasters found. expected {self.monthly_source_count}."
                )

    def read_and_sum_source_rasters(self):
        """Read the annual rasters and sum them into a single array."""
        annual_arr_list = []
        for raster_path in self.annual_raster_list:
            with rasterio.open(raster_path) as src:
                arr_data = src.read()
                annual_arr_list.append(arr_data)

        if len(annual_arr_list) > 1:
            self.annual_group_arr = np.nansum(annual_arr_list, axis=0)
        else:
            self.annual_group_arr = annual_arr_list[0]

        self.annual_group_arr = np.flip(self.annual_group_arr, axis=1)
        self.gridded_yearly_sum = np.nansum(self.annual_group_arr, axis=(1, 2))

        self.annual_mass_da = xr.DataArray(
            self.annual_group_arr,
            dims=["time", "y", "x"],
            coords={
                "time": years,
                "y": self.gepa_profile.y,
                "x": self.gepa_profile.x,
            },
            name=self.group_name,
        )

    def QC_group_grid(self):
        self.emi_check_df = self.emi_group_year_df.assign(
            gridded_emissions=self.gridded_yearly_sum
        ).assign(
            yearly_dif=lambda df: df["ghgi_ch4_kt"] - df["gridded_emissions"],
            rel_diff=lambda df: np.abs(
                (df["ghgi_ch4_kt"] - df["gridded_emissions"])
                / ((df["ghgi_ch4_kt"] + df["gridded_emissions"]) / 2)
            ),
            qc_pass=lambda df: (df["rel_diff"] < self.relative_tolerance),
            isclose_pass=lambda df: np.isclose(
                df["ghgi_ch4_kt"], df["gridded_emissions"], atol=0.0, rtol=0.0001
            ),
        )
        self.emi_check_df = self.emi_check_df.merge(
            self.source_qc_df, on="year"
        ).assign(
            emi_eq=lambda df: np.isclose(
                df["ghgi_ch4_kt"], df["source_emi_sum"], atol=0.0, rtol=0.0001
            ),
            grid_eq=lambda df: np.isclose(
                df["gridded_emissions"], df["source_grid_sum"], atol=0.0, rtol=0.0001
            ),
        )

        self.emi_check_df.to_csv(
            self.qc_dir / f"{self.group_name}_ch4_v3_emi_qc.csv", index=False
        )

        if not all(self.emi_check_df["qc_pass"]):
            print(f"\tare all emis eq: {self.emi_check_df["emi_eq"].all()}")
            print(f"\tare all grid eq: {self.emi_check_df["grid_eq"].all()}")
            raise ValueError("QC FAILED")
        else:
            print(
                f"QC PASSED: v3 gridded emissions for {self.group_name} "
                "match inventory emissions."
            )

        self._plot_v3_emission_check()

    def _plot_v3_emission_check(self) -> None:
        fig, (ax1, ax2) = plt.subplots(
            2, layout="constrained", figsize=(10, 6), dpi=300
        )
        fig.suptitle(
            f"{self.group_name}\n"
            "Comparison of v3 Inventory Emissions and Gridded Emissions",
            fontsize=14,
        )

        ax1.set_xlabel("Year")
        ax1.set(title="Inventory Emissions")
        ax1.set_ylabel("Emissions (kt)")
        ax1.plot(
            self.emi_check_df["year"],
            self.emi_check_df["ghgi_ch4_kt"],
            color="xkcd:violet",
        )

        ax2.plot(
            self.emi_check_df["year"],
            self.emi_check_df["rel_diff"],
            color="xkcd:green",
        )
        ax2.axhline(0, color="xkcd:slate", linestyle="--")
        ax2.set(title="Relative difference between inventory and gridded emissions")
        ax2.set_ylabel("Relative Difference (%)")

        plt.savefig(
            self.qc_dir / f"{self.group_name}_ch4_v3_emi_qc.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

    def plot_annual_raster_data(self) -> None:
        """
        Function to plot the raster data for each year in the dictionary of rasters that are
        output at the end of each sector script.
        """
        fg = self.annual_flux_da.where(lambda x: x > 0).plot.imshow(
            col="time",
            col_wrap=3,
            cmap=self.emi_custom_colormap,
            transform=ccrs.PlateCarree(),  # remember to provide this!
            subplot_kws={"projection": ccrs.PlateCarree()},
            cbar_kwargs={
                "orientation": "horizontal",
                "shrink": 0.8,
                "aspect": 40,
                "extend": "neither",
                "label": "methane emissions (Mg a$^{-1}$ km$^{-2}$)",
            },
            robust=True,
            figsize=(20, 20),
        )
        for ax in fg.axs.ravel():
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES)
            ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())

        fg.fig.suptitle(
            f"{self.group_name}\nGridded methane flux emissions", fontsize=14
        )

        # Save the plots as PNG files to the figures directory
        plt.savefig(self.qc_dir / f"{self.group_name}_ch4_v3_annual_flux.png")
        # Show the plot for review
        plt.show()
        # close the plot
        plt.close()

    def plot_raster_data_difference(self) -> None:
        """
        Function to plot the difference between the first and last years of the raster data
        for each sector.
        """
        # Define the geographic transformation parameters

        # Get the first and last years of the data
        list_of_data_years = list(self.annual_flux_da.time.values)

        first_year = np.min(list_of_data_years)
        last_year = np.max(list_of_data_years)

        first_year_data = self.annual_flux_da.sel(time=first_year)
        last_year_data = self.annual_flux_da.sel(time=last_year)

        # Calculate the difference between the first and last years
        self.difference_raster = (last_year_data - first_year_data).where(
            lambda x: x > 0
        )

        c_map, c_norm = self._get_cmap(self.difference_raster)
        fg = self.difference_raster.plot(
            cmap=c_map,
            transform=ccrs.PlateCarree(),  # remember to provide this!
            subplot_kws={"projection": ccrs.PlateCarree()},
            cbar_kwargs={
                "orientation": "horizontal",
                "shrink": 0.8,
                "aspect": 40,
                "norm": c_norm,
                "extend": "neither",
                "label": "Difference in methane emissions (Mg a$^{-1}$ km$^{-2}$)",
            },
            robust=True,
            figsize=(20, 10),
        )
        fg.axes.add_feature(cfeature.LAND)
        fg.axes.add_feature(cfeature.OCEAN)
        fg.axes.add_feature(cfeature.COASTLINE)
        fg.axes.add_feature(cfeature.STATES)
        fg.axes.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())

        # Add a title
        difference_plot_title = (
            f"{self.group_name}: Difference between {first_year} and\n"
            f"{last_year} methane emissions"
        )
        fg.figure.suptitle(difference_plot_title, fontsize=14)

        # Save the plot as a PNG file
        plt.savefig(
            self.qc_dir
            / f"{self.group_name}_ch4_v3_flux_difference_{first_year}-{last_year}.png"
        )
        # Show the plot for review
        plt.show()
        # close the plot
        plt.close()

    def calc_conversion_factor(self, year_days: int, area_matrix: np.array) -> np.array:
        """calculate emissions in kt to flux"""
        return (
            10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix
        )

    def calculate_flux(self, in_ds, timestep, direction):
        """calculates flux for dictionary of total emissions year/array pairs"""
        self.area_matrix = load_area_matrix()

        if direction not in ["mass2flux", "flux2mass"]:
            raise ValueError(
                f"direction must be either 'mass2flux' or 'flux2mass', not {direction}"
            )
        if timestep not in ["year", "year_month"]:
            raise ValueError(
                f"timestep must be either 'year' or 'year_month', not {timestep}"
            )

        times = in_ds.time.values

        def get_days_in_year(year):
            year = int(year)
            return 366 if calendar.isleap(year) else 365

        def get_days_in_month(year_month):
            year, month = year_month.split("-")
            year = int(year)
            month = int(month)
            return calendar.monthrange(year, month)[1]

        if timestep == "year_month":
            days_in_months = [get_days_in_month(x) for x in times]
            conv_factors = [
                self.calc_conversion_factor(x, self.area_matrix) for x in days_in_months
            ]

            conv_ds = xr.DataArray(
                conv_factors,
                # np.flip(conv_factors, 1),
                dims=["time", "y", "x"],
                coords=[times, self.gepa_profile.y, self.gepa_profile.x],
                name="conversion_factor",
            )
            if direction == "mass2flux":
                self.monthly_flux_da = in_ds * conv_ds
            elif direction == "flux2mass":
                self.monthly_flux_da = in_ds / conv_ds
        elif timestep == "year":
            days_in_year = [get_days_in_year(x) for x in times]
            conv_factors = [
                self.calc_conversion_factor(x, self.area_matrix) for x in days_in_year
            ]

            conv_ds = xr.DataArray(
                conv_factors,
                # np.flip(conv_factors, 1),
                dims=["time", "y", "x"],
                coords=[times, self.gepa_profile.y, self.gepa_profile.x],
                name="conversion_factor",
            )
            if direction == "mass2flux":
                flux_out_da = in_ds * conv_ds
            elif direction == "flux2mass":
                flux_out_da = in_ds / conv_ds
            return flux_out_da

    def QC_flux_emis(self) -> None:
        """
        Function to compare and plot the difference between v2 and v3 for each year of the
        raster data for each sector.
        """

        # Plot the difference between v2 and v3 methane emissions for each year

        actual_years = set(self.annual_flux_da.time.values.astype(int))
        expected_years = set(range(min_year, max_year + 1))
        missing_years = expected_years - actual_years
        if missing_years:
            Warning(
                f"Missing years in v3 data for {self.group_name}: {sorted(missing_years)}"
            )

        # Check for negative values
        for year in self.annual_flux_da.time:
            year_val = int(year.values)
            # v3_arr = np.flip(self.annual_flux_da.sel(time=year).values, 0)
            v3_arr = self.annual_flux_da.sel(time=year).values
            neg_count = np.sum(v3_arr < 0)
            if neg_count > 0:
                neg_percent = (neg_count / v3_arr.size) * 100
                Warning(
                    f"Source {self.group_name}, Year {year_val} has {neg_count} "
                    "negative values ({neg_percent:.2f}% of cells)"
                )

        # compare against v2 values, if v2_data exists
        if self.v2_name == "":
            Warning(
                f"there is no v2 raster data to compare against v3 for {self.group_name}!"
            )
        else:
            # Get v2 flux raster data
            v2_data_paths = V3_DATA_PATH.glob("Gridded_GHGI_Methane_v2_*.nc")
            v2_data_paths = [
                f for f in v2_data_paths if "Monthly_Scale_Factors" not in f.stem
            ]
            v2_data_dict = {}
            # The v2 data are not projected, so we get warnings reading all these files
            # suppress the warnings about no georeference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for in_path in v2_data_paths:
                    v2_year = int(in_path.stem.split("_")[-1])
                    v2_data = rioxarray.open_rasterio(in_path, variable=self.v2_name)[
                        self.v2_name
                    ].values.squeeze(axis=0)
                    v2_data_dict[v2_year] = v2_data

            v2_arr = np.array(list(v2_data_dict.values()))

            self.v2_flux_da = xr.DataArray(
                v2_arr,
                dims=["time", "y", "x"],
                coords=[
                    list(v2_data_dict.keys()),
                    self.gepa_profile.y,
                    self.gepa_profile.x,
                ],
                name=self.v2_name,
            )

            v3_time_match_da = self.annual_flux_da.sel(time=list(v2_data_dict.keys()))

            v2_flux_yearly_sums = np.nansum(self.v2_flux_da.values, axis=(1, 2))
            v3_flux_yearly_sums = np.nansum(v3_time_match_da.values, axis=(1, 2))

            flux_dif_df = pd.DataFrame(
                {
                    "year": list(v2_data_dict.keys()),
                    "v2_sum": v2_flux_yearly_sums,
                    "v3_sum": v3_flux_yearly_sums,
                }
            ).assign(metric="flux")

            self.flux_diff_da = (v3_time_match_da - self.v2_flux_da).where(
                lambda x: x != 0
            )

            v3_mass_da = self.calculate_flux(
                v3_time_match_da, timestep="year", direction="flux2mass"
            )
            v2_mass_da = self.calculate_flux(
                self.v2_flux_da, timestep="year", direction="flux2mass"
            )
            self.mass_diff_da = (v3_mass_da - v2_mass_da).where(lambda x: x != 0)
            self.write_tif_output(
                self.mass_diff_da,
                self.qc_dir / f"{self.group_name}_ch4_v3_v2_mass_diff.tif",
            )
            self.write_tif_output(
                self.flux_diff_da,
                self.qc_dir / f"{self.group_name}_ch4_v3_v2_flux_diff.tif",
            )

            v2_mass_yearly_sums = np.nansum(v2_mass_da.values, axis=(1, 2))
            v3_mass_yearly_sums = np.nansum(v3_mass_da.values, axis=(1, 2))

            mass_dif_df = pd.DataFrame(
                {
                    "year": list(v2_data_dict.keys()),
                    "v2_sum": v2_mass_yearly_sums,
                    "v3_sum": v3_mass_yearly_sums,
                }
            ).assign(metric="mass")

            self.flux_qc_df = pd.concat([flux_dif_df, mass_dif_df], axis=0).assign(
                yearly_dif=lambda df: df["v3_sum"] - df["v2_sum"],
                rel_diff=lambda df: np.abs(df["v3_sum"] - df["v2_sum"])
                / ((df["v3_sum"] + df["v2_sum"]) / 2),
            )
            self.flux_qc_df

            self._plot_percent_dif_fig()
            self._plot_difference_histogram()
            self._plot_difference_map()

    def _plot_percent_dif_fig(self):
        g = sns.relplot(
            kind="line",
            data=self.flux_qc_df,
            x="year",
            y="rel_diff",
            hue="metric",
            palette=sns.color_palette(["xkcd:violet", "xkcd:green"], 2),
        )
        g.figure.suptitle(f"{self.group_name} v2 v v3 Relative difference", fontsize=14)
        g.set_axis_labels("Year", "relative difference")
        g.savefig(self.qc_dir / f"{self.group_name}_ch4_v3_v2_percent_difference.png")
        plt.show()
        plt.close()

    def _plot_difference_map(self):
        c_map, c_norm = self._get_cmap(self.flux_diff_da)
        fg = self.flux_diff_da.plot.imshow(
            col="time",
            col_wrap=3,
            cmap=c_map,
            transform=ccrs.PlateCarree(),  # remember to provide this!
            subplot_kws={"projection": ccrs.PlateCarree()},
            cbar_kwargs={
                "orientation": "horizontal",
                "shrink": 0.8,
                "aspect": 40,
                "norm": c_norm,
                "extend": "neither",
                "label": "Difference in methane emissions (Mg a$^{-1}$ km$^{-2}$)",
            },
            robust=True,
            figsize=(20, 14),
        )
        for ax in fg.axs.ravel():
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES)
            ax.set_extent([-125, -66.5, 24, 49.5], crs=ccrs.PlateCarree())
        fg.fig.suptitle(f"{self.group_name} v2 to v3 methane emissions difference")
        # Add labels to the color bar
        cbar = fg.cbar
        cbar.ax.text(
            -0.01,
            0.5,
            "v2 higher",
            va="center",
            ha="right",
            transform=cbar.ax.transAxes,
        )
        cbar.ax.text(
            1.01, 0.5, "v3 higher", va="center", ha="left", transform=cbar.ax.transAxes
        )
        plt.savefig(
            self.qc_dir / f"{self.group_name}_ch4_v3_v2_flux_difference_maps.png"
        )
        plt.show()
        plt.close()

    def _plot_difference_histogram(self):
        """
        Function to plot a histogram of the differences between v2 and v3 fluxes for each year.
        This is useful for visualizing the distribution of differences in flux values.
        """
        fig, axs = plt.subplots(
            3, 3, figsize=(20, 12), sharex=True, sharey=True, layout="constrained"
        )
        for year, ax in zip(self.flux_diff_da.time.values, axs.ravel()):
            annual_flux = (
                self.annual_flux_da.sel(time=year)
                .where(lambda x: x > 0)
                .values.flatten()
            )
            v2_flux = (
                self.v2_flux_da.sel(time=year).where(lambda x: x > 0).values.flatten()
            )

            # Remove NaN values
            annual_flux = annual_flux[~np.isnan(annual_flux)]
            v2_flux = v2_flux[~np.isnan(v2_flux)]
            bins = np.histogram_bin_edges(
                np.concatenate([annual_flux, v2_flux]), bins=75
            )

            # Create histogram
            annual_flux, _ = np.histogram(annual_flux, bins)
            v2_flux, _ = np.histogram(v2_flux, bins)

            width = np.diff(bins)[0] * 0.4  # Adjust width for better visibility

            ax.bar(
                bins[:-1] + width / 2,
                annual_flux,
                width=width,
                label="v3 Flux",
                align="edge",
                color="xkcd:violet",
            )
            ax.bar(
                bins[:-1] - width / 2,
                v2_flux,
                width=width,
                label="V2 Flux",
                align="edge",
                color="xkcd:green",
            )

            ax.legend(loc="upper right", fontsize=8)
            ax.set(title=f"{int(year)}", yscale="log")
            plt.xlabel("Flux Value")
            plt.ylabel("Frequency")

        for ax in axs.ravel()[-2:]:
            ax.set_visible(False)
        fig.suptitle(f"{self.group_name} v2/v3 Flux Comparison", fontsize=16)
        plt.savefig(self.qc_dir / f"{self.group_name}_ch4_v3_v2_flux_histogram.png")
        plt.show()
        plt.close()

    def _get_cmap(self, in_da):
        # Define the colormap normalization. This depends on if there are values
        # greater than or less than 0. If they are all greater, use the reds colormap,
        # if they are all less than 0, use the blues colormap. If there are both,
        # use a two-slope normalization with a center at 0.
        # print(f"c_min: {c_min}, c_max: {c_max}")

        c_min = np.nanmin(in_da.values)
        c_max = np.nanmax(in_da.values)
        if c_min >= 0:
            c_norm = colors.Normalize(vmin=0, vmax=c_max)
            c_map = "Reds"
        elif c_max <= 0:
            c_norm = colors.Normalize(vmin=c_min, vmax=0)
            c_map = "Blues"
        else:
            c_norm = TwoSlopeNorm(vmin=c_min, vcenter=0, vmax=c_max)
            c_map = "coolwarm"

        return c_map, c_norm

    def calculate_monthly_scaling(self):
        """read all the monthly data, calculate a 3d array of monthly emissions
        and normalized it by year to sum to 12 for each year"""
        monthly_raster_ds_list = []
        for monthly_raster in self.monthly_raster_list:
            monthly_raster_ds_list.append(xr.open_dataset(monthly_raster))

        self.month_scale_ds = (
            xr.concat(monthly_raster_ds_list, dim="source")
            .sum(dim="source")
            .assign_coords(year=("band", np.repeat(np.arange(2012, 2023), 12)))
            .groupby(["year"])
            .apply(lambda x: (x / x.sum(dim="band")) * 12)
        )

        self.month_scale_check = (
            self.month_scale_ds.groupby("year")
            .sum()
            .where(lambda x: x > 0)
            .to_dataframe()
            .reset_index()
            .dropna(subset="band_data")
            .assign(sum_check=lambda df: np.isclose(df.band_data, 12, rtol=0.1))
        )
        self.month_scale_check["sum_check"].all()
        self.write_tif_output(self.month_scale_ds["band_data"], self.monthly_scale_output_path)

    def run_gridding(self):
        self.get_source_QC_df()
        self.get_group_emi_df()
        self.get_input_raster_paths()
        self.read_and_sum_source_rasters()
        self.QC_group_grid()
        self.annual_flux_da = self.calculate_flux(
            self.annual_mass_da, "year", "mass2flux"
        )
        self.QC_flux_emis()
        self.write_tif_output(self.annual_flux_da, self.tif_flux_output_path)
        self.write_tif_output(self.annual_mass_da, self.tif_kt_output_path)
        self.plot_annual_raster_data()
        self.plot_raster_data_difference()
        if self.monthly_source_count > 0:
            self.calculate_monthly_scaling()
