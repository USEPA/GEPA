import logging
import sqlite3

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import xarray as xr
from rasterio.features import rasterize
from tqdm.auto import tqdm
from joblib import Parallel, delayed


from gch4i.config import (
    emi_data_dir_path,
    gridded_output_dir,
    proxy_data_dir_path,
    global_data_dir_path,
    years,
)
from gch4i.utils import GEPA_spatial_profile, get_cell_gdf, normalize

logger = logging.getLogger(__name__)


def get_status_table(status_db_path, working_dir, the_date):
    # get a numan readable version of the status database
    conn = sqlite3.connect(status_db_path)
    status_df = pd.read_sql_query("SELECT * FROM gridding_status", conn)
    conn.close()
    status_df.to_csv(working_dir / f"gridding_status_{the_date}.csv", index=False)
    return status_df


class EmiProxyGridder:

    def __init__(self, emi_proxy_in_data, db_path, qc_dir):
        self.state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
        self.county_geo_path = global_data_dir_path / "tl_2020_us_county.zip"
        self.get_geo_filter()
        self.qc_dir = qc_dir
        self.file_type = emi_proxy_in_data.file_type
        self.emi_time_step = emi_proxy_in_data.emi_time_step
        self.emi_geo_level = emi_proxy_in_data.emi_geo_level
        self.gch4i_name = emi_proxy_in_data.gch4i_name
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
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.get_status()
        if self.status is None:
            self.status = "not started"
            self.update_status()
        self.base_name = f"{self.gch4i_name}-{self.emi_id}-{self.proxy_id}"
        self.emi_input_path = list(emi_data_dir_path.glob(f"{self.emi_id}.csv"))[0]
        self.proxy_input_path = list(proxy_data_dir_path.glob(f"{self.proxy_id}.*"))[0]
        self.annual_output_path = gridded_output_dir / f"{self.base_name}.tif"
        self.has_monthly = (
            self.emi_time_step == "monthly" or self.proxy_time_step == "monthly"
        )
        if self.has_monthly:
            self.monthly_output_path = (
                gridded_output_dir / f"{self.base_name}_monthly.tif"
            )
        else:
            self.monthly_output_path = None
        self.time_col = self.get_time_col()
        self.gepa_profile = GEPA_spatial_profile()
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

    def get_geo_filter(self):
        self.get_state_gdf()
        self.geo_filter = self.state_gdf.state_code.unique().tolist() + ["OF"]

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
        else:
            self.proxy_rel_emi_col = self.proxy_rel_emi_col

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
                self.femi_df = pd.read_csv(
                    self.emi_input_path,
                    usecols=self.emi_cols,
                ).query("(state_code.isin(@state_gdf.state_code)) & (ghgi_ch4_kt > 0)")
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

        self.time_geo_qc_df.to_csv(self.qc_dir / f"{self.base_name}_qc_state_year.csv")

        if not self.time_geo_qc_df.has_proxy.all():
            if self.geo_col:
                failed_states = (
                    self.time_geo_qc_df[self.time_geo_qc_df["has_proxy"] == False]
                    .groupby(self.geo_col)
                    .size()
                )
                logging.critical(
                    f"QC FAILED: {self.emi_id}, {self.proxy_id}\n"
                    "proxy state/year columns do not match emissions\n"
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
            logging.info(f"QC PASS: state/year QC.")
        else:
            logging.critical(f"QC FAIL: state/year QC.\n")
            self.status = "failed state/year QC"
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

        monthly_scaling = (
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
        monthly_scaling

        check_scaling = (
            monthly_scaling.reset_index()
            .assign(year=lambda df: df["year_month"].str.split("-").str[0])
            .groupby(["state_code", "year"])["month_normed"]
            .sum()
            .to_frame()
            .assign(isclose=lambda df: np.isclose(df["month_normed"], 1))
        )
        check_scaling["isclose"].all()

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
                monthly_scaling,
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

    def QC_proxy_allocation(self, plot=False, plot_path=None) -> pd.DataFrame:
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
                self.proxy_ds = self.grid_allocated_emissions(self.allocation_gdf)
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
            print(time_var)

            # orig_emi_val = data["allocated_ch4_kt"].sum()
            # if we need to disaggregate non-point data. Maybe extra, but this accounts for
            # potential edge cases where the data change from year to year.

            if not (time_data.geometry.type == "Points").all():
                data_to_concat = []
                for geom_type, geom_data in time_data.groupby(time_data.geom_type):
                    print(geom_type)
                    # regular points can just be passed on "as is"
                    if geom_type == "Point":
                        # get the point data
                        print("doing point data")
                        point_data = geom_data.loc[:, ["geometry", "allocated_ch4_kt"]]
                        print(point_data.is_empty.any())
                        data_to_concat.append(point_data)
                    # if we have multipoint data, we need to disaggregate the emissions
                    # across the individual points and align them with the cells
                    elif geom_type == "MultiPoint":
                        print("doing multipoint data")
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
                        print(f" any empty data: {multi_point_data.is_empty.any()}")
                        data_to_concat.append(multi_point_data)
                    # if the data are any kind of lines, compute the relative length within
                    # each cell
                    elif "Line" in geom_type:
                        print("doing line data")
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
                        print(f" any empty data: {line_data.is_empty.any()}")
                        data_to_concat.append(line_data)
                    # if the data are any kind of  polygons, compute the relative area in
                    # each cell
                    elif "Polygon" in geom_type:
                        print("doing polygon data")
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
                        print(f" any empty data: {polygon_data.is_empty.any()}")
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

            print("compare pre and post processing sums:")
            print("pre data:  ", time_data[["allocated_ch4_kt"]].sum())
            print("post data: ", ready_data[["allocated_ch4_kt"]].sum())

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
        self.check_vector_proxy_time_geo("year")
        if (self.proxy_time_step == "monthly") & (self.emi_time_step == "annual"):
            self.scale_emi_to_month()
        self.allocate_emissions_to_proxy()
        self.QC_proxy_allocation()
        self.grid_allocated_emissions()

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
            emi_gdf = adm_gdf.merge(e_df, on="fips")
            emi_raster = rasterize(
                [
                    (geom, value)
                    for geom, value in zip(emi_gdf.geometry, emi_gdf.ghgi_ch4_kt)
                ],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype="float32",
                merge_alg=rasterio.enums.MergeAlg.add,
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
            grouper_cols = [self.geo_col, self.time_col]
        else:
            grouper_cols = [self.time_col]

        if self.time_col == "year_month":
            self.proxy_ds = self.proxy_ds.assign_coords(
                year_month=pd.to_datetime(
                    pd.DataFrame(
                        {
                            "year": self.proxy_ds.year.values,
                            "month": self.proxy_ds.month.values,
                            "day": 1,
                        }
                    )
                ).dt.strftime("%Y-%m")
            )

        rel_emi_check = (
            self.proxy_ds["rel_emi"]
            .groupby(grouper_cols)
            .sum()
            .to_dataframe()
            .reset_index()
            .set_index(grouper_cols)
        )
        final_df = self.emi_df.set_index(grouper_cols).join(rel_emi_check)
        self.time_geo_qc_df = final_df[
            (final_df.ghgi_ch4_kt > 0) & (final_df.rel_emi < 0.9)
        ]
        self.time_geo_qc_df.to_csv(self.qc_dir / f"{self.base_name}_qc_state_year.csv")
        if len(self.time_geo_qc_df) > 0:
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

        # if the emi is month and the proxy is annual, we expand the dimensions of
        # the proxy, repeating the year values for every month in the year
        # we stack the year/month dimensions into a single year_month so that
        # it aligns with the emissions data as a time x X x Y array.
        if self.emi_time_step == "monthly" and self.proxy_time_step == "annual":
            self.proxy_ds = (
                self.proxy_ds.expand_dims(dim={"month": np.arange(1, 13)}, axis=0)
                .stack({"year_month": ["year", "month"]})
                .sortby(["year_month", "y", "x"])
            )

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

    def QC_emi_raster_sums(self, QC_time_col):
        """compares yearly array sums to inventory emissions"""

        emi_sum_check = (
            self.emi_df.groupby(self.time_col)["ghgi_ch4_kt"].sum().to_frame()
        )
        proxy_sum_check = self.proxy_ds.sum(dim=["x", "y"]).to_dataframe()

        if QC_time_col == "year_month" and isinstance(
            proxy_sum_check.index, pd.MultiIndex
        ):
            proxy_sum_check.index = proxy_sum_check.index.map(
                lambda x: f"{x[0]}-{x[1]:02d}"
            )
            relative_tolerance = 0.003
        else:
            relative_tolerance = 0.0001

        # The relative tolerance is the maximum allowable difference between the two values
        # as a fraction of the average of the two values. It is used to determine if
        # the two values are close enough to be considered equal.
        # The default value is 1e-5, which means that the two values must be within 0.01%
        # of each other to be considered equal.
        raster_qc_df = emi_sum_check.join(proxy_sum_check).assign(
            isclose=lambda df: np.isclose(df["ghgi_ch4_kt"], df["results"], rtol=1e-5),
            rel_diff=lambda df: np.abs(df["ghgi_ch4_kt"] - df["results"])
            / ((df["ghgi_ch4_kt"] + df["results"]) / 2),
            qc_pass=lambda df: (df["rel_diff"] < relative_tolerance),
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
        out_data = np.flip(in_ds.values, axis=1)
        out_profile.update(count=out_data.shape[0])
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(out_data)

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
