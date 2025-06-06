"""
Name:                   task_aog_wells_proxy.py
Date Last Modified:     2025-06-05
Authors Name:           A. Burnette, Nick Kruskamp (RTI International)
Purpose:                Mapping of abandoned wells oil/gas proxy emissions
Input Files:            - Lat/Lon: NEI_Reference_Grid_LCC_to_WGS84_latlon.shp
                        - Enverus Path: DIDSK_HEADERS_API10_2019_abandoned_wells.csv
                        - NEI Input (prefix): CONUS_SA_FILES_
Output Files:           - aog_gas_wells_proxy.parquet
                        - aog_oil_wells_proxy.parquet
Notes:                  - Indiana and Illinois data from Enverus Path is inaccurate and
                        was corrected in V2 with NEI data, with 2018 coming from a
                        separate MS Access source. V3 has updated NEI data for 2018-2022
                        and was used to correct V3.
                        - Slight differences in outputs occur from V2 to V3 due to the
                        switch to geopandas and the use of centroid points for NEI data
                        (2018-2022). Enverus and NEI data was joined to state_code
                        based on geometry, whereas V2 used lat/lon gridding.
                        - Three states have emi data that do not have proxy data:
                            - FL, 2012-2022, GAS
                            - ID, 2012-2022, OIL
                            - MD, 2012-2022, OIL
                        Proxy data was created for these states by assigning rel_emi = 1
                        and distributing emissions evenly across the state. NOTE: Next
                        version should consider evaluting NEI data as source of proxy
                        data for these states.
"""

########################################################################################
# %% Load Packages

from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import pytask
from pytask import Product

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
)
from gch4i.config import years as YEARS

########################################################################################


# %%


class AOGWellsProxy:

    def __init__(
        self,
        aban_wells_path,
        state_input_path,
        nei_grid_path,
        well_type,
        ERG_NEI_input,
        file_extension,
        emi_path,
        output_path,
    ):
        self.aban_wells_path = aban_wells_path
        self.state_input_path = state_input_path
        self.nei_grid_path = nei_grid_path
        self.well_type = well_type
        self.ERG_NEI_input = ERG_NEI_input
        self.years = YEARS
        self.file_extension = file_extension
        self.emi_path = emi_path
        self.output_path = output_path

    def read_state_data(self):
        # Read in state data
        self.state_gdf = (
            gpd.read_file(self.state_input_path)
            .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
            .rename(columns=str.lower)
            .rename(columns={"stusps": "state_code", "name": "state_name"})
            .astype({"statefp": int})
            # get only lower 48 + DC
            .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
            .to_crs(4326)
        )

    def read_nei_grid_data(self):
        # Read in shape file for Lat/Lon
        self.nei_grid_gdf = gpd.read_file(self.nei_grid_path).assign(
            ROW=lambda x: x["cellid"].str.split("!").str[1].astype(int),
            COL=lambda x: x["cellid"].str.split("!").str[0].astype(int),
        )

    def read_abandoned_well_data(self):
        # Read in Enverus Abandoned Well Data (from ERG)
        location_data = (
            pd.read_csv(
                self.aban_wells_path,
                low_memory=False,
                usecols=[
                    "API10",
                    "STATE",
                    "PRODUCING_ENTITY_COUNT",
                    "SURFACE_HOLE_LATITUDE_WGS84",
                    "SURFACE_HOLE_LONGITUDE_WGS84",
                    "LAST_PROD_DATE",
                    "COMPLETION_DATE",
                    "SPUD_DATE",
                    "ABANDONED_WELL_TYPE",
                    "PLUGGED_UNPLUGGED",
                ],
            )
            .rename(
                {
                    "SURFACE_HOLE_LATITUDE_WGS84": "lat",
                    "SURFACE_HOLE_LONGITUDE_WGS84": "lon",
                },
                axis=1,
            )
            .rename(columns=lambda x: str(x).lower())
            .query("state in @self.state_gdf.state_code")
            .query(f"abandoned_well_type.isin(['{self.well_type}', 'dry'])")
            .assign(
                prod_year=lambda x: pd.to_datetime(x["last_prod_date"]).dt.year,
                comp_year=lambda x: pd.to_datetime(x["completion_date"]).dt.year,
                spud_year=lambda x: pd.to_datetime(x["spud_date"]).dt.year,
                plugged_unplugged=lambda df: df.plugged_unplugged.str.casefold(),
            )
        )

        # location_data_corr: Convert and Filter location_data to only include lower 48 + DC
        location_data_corr = gpd.GeoDataFrame(
            location_data,
            geometry=gpd.points_from_xy(location_data.lon, location_data.lat),
            crs=4326,
        )

        valid_geoms = location_data_corr.is_valid
        not_empty_geoms = ~location_data_corr.is_empty

        self.bad_geom_well_gdf = location_data_corr[~valid_geoms | ~not_empty_geoms]
        print(f"there are {len(self.bad_geom_well_gdf )} bad geometries in the data")
        self.aban_well_gdf = location_data_corr[valid_geoms & not_empty_geoms]

    def get_abandoned_well_ratios(self):
        # Create a list to store results
        results = []

        # Iterate through each year
        for year in self.years:
            # Filter wells abandoned by the given year
            temp = self.aban_well_gdf[
                (self.aban_well_gdf["prod_year"] < year)
                | (
                    (self.aban_well_gdf["prod_year"].isna())
                    & (self.aban_well_gdf["comp_year"] < year)
                )
                | (
                    (self.aban_well_gdf["prod_year"].isna())
                    & (self.aban_well_gdf["comp_year"].isna())
                    & (self.aban_well_gdf["spud_year"] < year)
                )
            ]
            # Group by state and well type, sum producing entity counts
            well_counts = (
                temp.groupby(["state", "abandoned_well_type"])["producing_entity_count"]
                .sum()
                .unstack(fill_value=0)
            )

            # Ensure both 'GAS' and 'OIL' columns exist
            if "GAS" not in well_counts.columns:
                well_counts["GAS"] = 0
            if "OIL" not in well_counts.columns:
                well_counts["OIL"] = 0

            # Calculate the gas to oil ratio
            well_counts["gas_to_oil_ratio"] = well_counts["GAS"] / (
                well_counts["GAS"] + well_counts["OIL"]
            ).replace(0, np.nan)

            # Reset index to make state a column and add year
            year_results = well_counts.reset_index()
            year_results["year"] = year

            # Filter to only include states in state_list
            year_results = year_results[
                year_results["state"].isin(self.state_gdf["state_code"])
            ]

            results.append(year_results[["state", "year", "gas_to_oil_ratio"]])

        # Combine results into a single DataFrame
        self.abandoned_well_ratios = pd.concat(results, ignore_index=True).fillna(0)

    def get_base_results(self):
        # Create a list to store results
        results = []

        # Iterate through each year
        for year in self.years:
            # Filter wells abandoned by the given year
            temp = self.aban_well_gdf[
                (self.aban_well_gdf["prod_year"] < year)
                | (
                    (self.aban_well_gdf["prod_year"].isna())
                    & (self.aban_well_gdf["comp_year"] < year)
                )
                | (
                    (self.aban_well_gdf["prod_year"].isna())
                    & (self.aban_well_gdf["comp_year"].isna())
                    & (self.aban_well_gdf["spud_year"] < year)
                )
            ]
            # Add year to temp
            temp["year"] = year
            # Allocate Dry wells to either gas or oil
            temp = temp.merge(
                self.abandoned_well_ratios, on=["state", "year"], how="left"
            )

            # OPTION: sample the dry wells according to the state gas to oil ratio for
            # the gas or oil proxy
            # temp.query('abandoned_well_type == "DRY"').sample(frac=gas_to_oil_ratio, random_state=42)

            # Create Gas portion of Dry wells
            temp_gas = temp.query('abandoned_well_type == "DRY"').assign(
                abandoned_well_type="GAS",
                producing_entity_count=lambda x: x["producing_entity_count"]
                * x["gas_to_oil_ratio"],
            )
            # Create Oil portion of Dry wells
            temp_oil = temp.query('abandoned_well_type == "DRY"').assign(
                abandoned_well_type="OIL",
                producing_entity_count=lambda x: x["producing_entity_count"]
                * (1 - x["gas_to_oil_ratio"]),
            )
            # Join Gas and Oil portions with original data
            temp = temp.query('abandoned_well_type != "DRY"')
            temp = (
                pd.concat([temp, temp_gas, temp_oil], ignore_index=True)
                .rename(columns={"state": "state_code"})
                .loc[
                    :,
                    [
                        "state_code",
                        "year",
                        "producing_entity_count",
                        "abandoned_well_type",
                        "geometry",
                        "plugged_unplugged",
                    ],
                ]
            )
            # Append to results
            results.append(temp)

        # Remove IL and IN data. It is inaccurate and will be fixed in the next step
        self.base_results = (
            pd.concat(results, ignore_index=True).query(
                'state_code != "IL" & state_code != "IN"'
            )
            # Query for filter condition in abandoned_well_type
            .query(f"abandoned_well_type == '{self.well_type}'")
        )

        self._scale_data_plug_status()

    def get_IL_IN_data(self):
        # Initialize results
        IL_IN_adj_list = []

        # Determine year data based on report_year
        self.report_year_crosswalk = {
            2012: 2011,
            2013: 2014,
            2014: 2014,
            2015: 2014,
            2016: 2017,
            2017: 2017,
            2018: 2018,
            2019: 2019,
            2020: 2020,
            2021: 2021,
            2022: 2022,
        }

        # Iterate through each year
        for proxy_year in self.years:

            # Get report year from crosswalk
            report_year = self.report_year_crosswalk[proxy_year]

            print(f"proxy year: {proxy_year}, report year: {report_year}")

            # Clean table based on year
            # data_temp: NEI Data
            if report_year >= 2011 and report_year <= 2017:
                path = f"{ERG_NEI_input}{report_year}{self.file_extension}"
                print(f"does the path exist:              {Path(path).exists()}")
                print(f"is the right year in the path:    {str(report_year) in path}")
                data_temp = pd.read_csv(path, sep="\t", skiprows=25)
                data_temp = data_temp.drop(["!"], axis=1)
                data_temp.columns = [
                    "Code",
                    "FIPS",
                    "COL",
                    "ROW",
                    "Frac",
                    "Abs",
                    "FIPS_Total",
                    "FIPS_Running_Sum",
                ]

                # Merge NEI data with shape data
                temp_result = (
                    data_temp.merge(self.nei_grid_gdf, on=["COL", "ROW"], how="left")
                    .filter(["Abs", "Latitude", "Longitude"])
                    .rename(columns={"Abs": "producing_entity_count"})
                )

                # Convert to GeoDataFrame
                temp_result = gpd.GeoDataFrame(
                    temp_result,
                    geometry=gpd.points_from_xy(
                        temp_result["Longitude"], temp_result["Latitude"]
                    ),
                    crs=4326,
                ).drop(columns=["Longitude", "Latitude"])

                # Merge with state data
                temp_result = (
                    gpd.sjoin(
                        temp_result,
                        self.state_gdf[["state_code", "geometry"]],
                        how="left",
                        predicate="within",
                    ).drop(columns="index_right")
                    # Filter for only IL and IN data
                    .query('state_code == "IL" | state_code == "IN"')
                    # Assign year and abandoned_well_type
                    .assign(year=proxy_year, abandoned_well_type=self.well_type)
                )
                # Append to IL_IN_adj
                IL_IN_adj_list.append(temp_result)

            # If year is 2018-2022, data is in shapefile format
            elif report_year >= 2018 and report_year <= 2022:
                # XXX: What I've found is that the code as it was written was not reading
                # the correct file for certain years. This is going to need to be fixed.
                # E.G. As it was written before, the file for 2018 does not exist, but it
                # was not previously throwing an error. So that tells me it was pulling from
                # the previous path that was already in mem.
                if report_year == 2020:
                    path = f"{self.ERG_NEI_input}{report_year}/{self.well_type}_WELL.shp"
                elif report_year == 2021:
                    if self.well_type == "GAS":
                        path = f"{self.ERG_NEI_input}{report_year}/_698.shp"
                    else:
                        path = f"{self.ERG_NEI_input}{report_year}/_695.shp"
                elif report_year == 2022:
                    if self.well_type == "GAS":
                        path = f"{self.ERG_NEI_input}{report_year}/GasWells.shp"
                    else:
                        path = f"{self.ERG_NEI_input}{report_year}/OilWells.shp"
                else:
                    path = f"{self.ERG_NEI_input}{report_year}/{self.well_type}_WELLS.shp"
                print(f"does the path exist:              {Path(path).exists()}")
                print(f"is the right year in the path:    {str(report_year) in path}")
                # Read in shapefile
                temp_result = gpd.read_file(path)
                # Convert crs to calculate centroid
                temp_result = (
                    temp_result.to_crs(3857)
                    .assign(geomtry=lambda df: df.centroid)
                    .to_crs(4326)
                    .sjoin(
                        self.state_gdf[["state_code", "geometry"]],
                    )
                    .drop(columns="index_right")
                    # Filter for only IL and IN data
                    .query('state_code == "IL" | state_code == "IN"')
                    # Assign year and abandoned_well_type
                    .assign(year=proxy_year, abandoned_well_type=self.well_type)
                )
                # Filter and rename columns based on year
                if report_year <= 2020:
                    temp_result = temp_result.filter(
                        [
                            "ACTIVITY",
                            "state_code",
                            "geometry",
                            "year",
                            "abandoned_well_type",
                        ]
                    ).rename(columns={"ACTIVITY": "producing_entity_count"})
                elif report_year == 2021:
                    temp_result = temp_result.filter(
                        [
                            "NUMBER_",
                            "state_code",
                            "geometry",
                            "year",
                            "abandoned_well_type",
                        ]
                    ).rename(columns={"NUMBER_": "producing_entity_count"})
                elif report_year == 2022:
                    temp_result = temp_result.filter(
                        [
                            "NUMBER_WEL",
                            "state_code",
                            "geometry",
                            "year",
                            "abandoned_well_type",
                        ]
                    ).rename(columns={"NUMBER_WEL": "producing_entity_count"})
                # If year is not in range, print error
                else:
                    print("MID: NEI DATA MISSING FOR YEAR ", report_year)
                # Append to IL_IN_adj
                IL_IN_adj_list.append(temp_result)
            # If year is not in range, print error
            else:
                print("END: NEI DATA MISSING FOR YEAR ", report_year)
            print()
        # Combine results into a single DataFrame
        self.IL_IN_adj_gdf = pd.concat(IL_IN_adj_list, ignore_index=True)

    def prepare_final_proxy(self):
        # Combine base_results and IL_IN_adj
        prelim_proxy_gdf = pd.concat(
            [self.base_results, self.IL_IN_adj_gdf], ignore_index=True
        )

        """
        Remove empty state/year/well_type combinations.
        This will prevent division by zero errors in the next step.
        This will enable checking for missing proxy data for emi data
        """

        # Remove empty proxies
        prelim_proxy_gdf = (
            prelim_proxy_gdf
            # Generate group sum for state_code, year, abandoned_well_type
            .assign(
                group_sum=lambda x: x.groupby(
                    ["state_code", "year", "abandoned_well_type"]
                )["producing_entity_count"].transform("sum")
            )
            # Filter out empty groups
            .query("group_sum != 0")
            # Drop group_sum column
            .drop(columns="group_sum")
        )

        """
        Emi Data exists, but no proxy data for these:
        FL, 2012-2022, GAS
        ID, 2012-2022, OIL
        MD, 2012-2022, OIL
        """

        ####################################################################################
        # STEP 6. Calculate grouped_proxy
        """
        Calculate the relative emissions for each state/year/well_type.
        The rel_emi will be used to allocate emissions to the CONUS region

        rel_emi = geometry[producing_entity_count]  / state_year_sum[producing_entity_count]
        """
        proxy_gdf = (
            prelim_proxy_gdf
            # Sum emissions
            .groupby(
                ["state_code", "year", "abandoned_well_type", "geometry"],
                as_index=False,
            )
            .agg({"producing_entity_count": "sum"})
            # Calculate state relative emissions (emissions / state_year emissions)
            .assign(
                state_year_sum=lambda x: x.groupby(
                    ["state_code", "year", "abandoned_well_type"]
                )["producing_entity_count"].transform("sum"),
                rel_emi=lambda x: x["producing_entity_count"] / x["state_year_sum"],
            )
            # Drop columns
            .drop(columns=["state_year_sum", "producing_entity_count"])
            # Set geometry and CRS
            .set_geometry("geometry")
            .set_crs("EPSG:4326")
        )

        ####################################################################################
        # STEP 7. Check for missing proxy data AND Create alternative proxy data
        """
        Steps:
            - Check if Proxy data is missing for a state/year/well_type
            - Create Alternative Proxy Data: rel_emi = 1, geometry = state polygon
                - This distributes emissions evenly across the state
        """
        # Build dictionary to map well to well emissions data
        proxy_dict = {"GAS": "aog_gas_wells_emi", "OIL": "aog_oil_wells_emi"}

        # Filter proxy_gdf to ensure only relevant proxy data
        filtered_proxy = proxy_gdf.query(f"abandoned_well_type == '{self.well_type}'")

        # Create filtered dictionary
        # filtered_dict = {well_type: proxy_dict[well_type]}

        # Check if proxy data exists for emissions data
        # for key, value in filtered_dict.items():
        emi_df = (
            pd.read_csv(self.emi_path)
            .query("ghgi_ch4_kt != 0")
            .query("state_code != 'AK'")
            .drop(columns=["Unnamed: 0"])
        )

        # Retrieve unique state codes for emissions without proxy data
        # This step is necessary, as not all emissions data excludes emission-less states
        emi_states = set(
            emi_df[["state_code", "year"]].itertuples(index=False, name=None)
        )
        proxy_states = set(
            filtered_proxy[["state_code", "year"]].itertuples(index=False, name=None)
        )

        # Find missing states
        missing_states = emi_states.difference(proxy_states)

        # Add missing states alternative data to grouped_proxy
        alt_proxy_list = []
        if missing_states:
            # Create alternative proxy from missing states
            alt_proxy = (
                pd.DataFrame(missing_states, columns=["state_code", "year"])
                # Assign well type and make rel_emi = 1
                .assign(abandoned_well_type=self.well_type, rel_emi=1)
                # Merge state polygon geometry
                .merge(
                    self.state_gdf[["state_code", "geometry"]],
                    on="state_code",
                    how="left",
                )
            )
            # Convert to GeoDataFrame
            alt_proxy = gpd.GeoDataFrame(
                alt_proxy, geometry="geometry", crs="EPSG:4326"
            )
            # Append to grouped_proxy
            alt_proxy_list.append(alt_proxy)

        missing_state_proxy = pd.concat(alt_proxy_list, ignore_index=True)

        proxy_gdf = pd.concat([proxy_gdf, missing_state_proxy], ignore_index=True)

        sum_check = (
            proxy_gdf.groupby(["state_code", "year"])["rel_emi"]
            .sum()
            .to_frame()
            .assign(qc_pass=lambda df: np.isclose(df["rel_emi"], 1, atol=0, rtol=1e-5))
        )
        if not sum_check["qc_pass"].all():
            raise ValueError(
                "Relative emissions do not sum to 1 for all state/year combinations."
            )
        else:
            print("proxy cleared state / year normalization check")
            proxy_gdf.to_parquet(self.output_path)

    def _scale_data_plug_status(self):
        """
        First piece of AOG logic:
        If state in {KY, NY, OH, PA. TN, WV} and status == plugged -> weight = 0.357
        If state in {KY, NY, OH, PA. TN, WV} and status == unplugged -> weight = 30.57
        If other state and status == plugged -> weight = 0.002
        If other state and status == unplugged -> weight = 10.02
        """

        # First piece of AOG logic:
        # If state in {KY, NY, OH, PA. TN, WV} and status == plugged -> weight = 0.357
        # If state in {KY, NY, OH, PA. TN, WV} and status == unplugged -> weight = 30.57
        # If other state and status == plugged -> weight = 0.002
        # If other state and status == unplugged -> weight = 10.02

        plug_state_list = ["KY", "NY", "OH", "PA", "TN", "WV"]

        plug_status_1 = (self.base_results.state_code.isin(plug_state_list)) & (
            self.base_results.plugged_unplugged == "plugged"
        )
        plug_status_2 = (self.base_results.state_code.isin(plug_state_list)) & (
            self.base_results.plugged_unplugged == "unplugged"
        )
        plug_status_3 = ~self.base_results.state_code.isin(plug_state_list) & (
            self.base_results.plugged_unplugged == "plugged"
        )
        plug_status_4 = ~self.base_results.state_code.isin(plug_state_list) & (
            self.base_results.plugged_unplugged == "unplugged"
        )

        self.base_results["weight"] = 0.0
        self.base_results.loc[plug_status_1, "weight"] = 0.357
        self.base_results.loc[plug_status_2, "weight"] = 30.57
        self.base_results.loc[plug_status_3, "weight"] = 0.002
        self.base_results.loc[plug_status_4, "weight"] = 10.02

        self.base_results["orig_producing_entity_count"] = self.base_results[
            "producing_entity_count"
        ]

        self.base_results["producing_entity_count"] = (
            self.base_results["producing_entity_count"] * self.base_results["weight"]
        )
        all_weight_vals = self.base_results["weight"].eq(0).any()
        print(f"are any weights still 0? {all_weight_vals}")

    def calc_national_plug_weights(self):
        """calc national plug weights

        Optional future use if we want to apply national plug weights to data that
        do not have individual well plug status."""
        self.nat_plug_weights_df = (
            self.base_results["plugged_unplugged"].value_counts(normalize=True).round(2)
        )

    def run_proxy_creation(self):
        """
        Run the proxy creation steps in order
        """
        self.read_state_data()
        self.read_nei_grid_data()
        self.read_abandoned_well_data()
        self.get_abandoned_well_ratios()
        self.get_base_results()
        self.get_IL_IN_data()
        self.prepare_final_proxy()


# %% Load Path Files & Repeated Variables


sector_path = sector_data_dir_path / "abandoned_aog_wells"

# State Path
state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"

# NEI grid path
nei_grid_path = (
    V3_DATA_PATH / "sector" / "nei_og" / "NEI_Reference_Grid_LCC_to_WGS84_latlon.shp"
)

# Enverus Path: Base Proxy Data
abadonded_wells_path = sector_path / "abandoned_wells.csv"

# Dir prefix for ERG NEI files
ERG_NEI_input = V3_DATA_PATH / "sector" / "nei_og" / "CONUS_SA_FILES_"

# Oil and Gas ERG NEI files (2012-2017)
# ERG_NEI_gas = "/USA_698_NOFILL.txt"
# ERG_NEI_oil = "/USA_695_NOFILL.txt"

param_dict = dict()

param_dict["oil"] = dict(
    emi_path=emi_data_dir_path / "aog_oil_wells_emi.csv",
    aban_wells_path=abadonded_wells_path,
    state_input_path=global_data_dir_path / "tl_2020_us_state.zip",
    nei_grid_path=nei_grid_path,
    well_type="OIL",
    ERG_NEI_input=ERG_NEI_input,
    file_extension="/USA_695_NOFILL.txt",
    output_path=proxy_data_dir_path / "aog_oil_wells_proxy.parquet",
)
param_dict["gas"] = dict(
    emi_path=emi_data_dir_path / "aog_gas_wells_emi.csv",
    aban_wells_path=abadonded_wells_path,
    state_input_path=global_data_dir_path / "tl_2020_us_state.zip",
    nei_grid_path=nei_grid_path,
    well_type="GAS",
    ERG_NEI_input=ERG_NEI_input,
    file_extension="/USA_698_NOFILL.txt",
    output_path=proxy_data_dir_path / "aog_gas_wells_proxy.parquet",
)
# %%


def task_aog_proxy_data(
    emi_path: Path,
    aban_wells_path: Path,
    state_input_path: Path,
    nei_grid_path: Path,
    well_type: str,
    ERG_NEI_input: Path,
    file_extension: str,
    output_path=Annotated[Path, Product],
):

    proxy_creator = AOGWellsProxy(
        aban_wells_path,
        state_input_path,
        nei_grid_path,
        well_type,
        ERG_NEI_input,
        file_extension,
        emi_path,
        output_path,
    )
    proxy_creator.run_proxy_creation()


# %%


sesh = pytask.build(
    tasks=[task_aog_proxy_data(**kwargs) for _id, kwargs in param_dict.items()]
)

# %%
