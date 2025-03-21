"""
Name:                   task_generators_proxy.py
Date Last Modified:     2025-03-21
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Mapping of natural gas generator proxy emissions
Input Files:            NG Storage Compressor Stations: proxy_data_dir_path / "ng_storage_comp_station_proxy.parquet"
                        NG Transmission Compressor Stations: proxy_data_dir_path / "ng_trans_comp_station_proxy.parquet"
                        GHGI: ghgi_data_dir_path / "1B2biv_ng_transmission_storage/NaturalGasSystems_90-22_FR.xlsx"
Output Files:           proxy_data_dir_path / "generators_proxy.parquet"
"""

# %%
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import numpy as np
from pytask import Product, mark, task

from gch4i.config import (
    max_year,
    min_year,
    years,
    ghgi_data_dir_path,
    proxy_data_dir_path,
    V3_DATA_PATH
)

# %%
@mark.persist
@task(id="generators_proxy")
def task_get_generators_proxy_data(
    ng_storage_comp_station_proxy_path: Path = proxy_data_dir_path / "ng_storage_comp_station_proxy.parquet",
    ng_trans_comp_station_proxy_path: Path = proxy_data_dir_path / "ng_trans_comp_station_proxy.parquet",
    ng_ghgi_path: Path = ghgi_data_dir_path / "1B2biv_ng_transmission_storage/NaturalGasSystems_90-22_FR.xlsx",
    generators_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "generators_proxy.parquet",
    ):
    """
    This proxy is the weighted average of the ng_storage_comp_station_proxy and the
    ng_trans_comp_station_proxy. Weighting is determined by calculating the fraction of
    generator emissions that occur at transmission vs. storage compressor stations.

    """

    # Read in the ng_storage_comp_station_proxy
    ng_storage_comp_station_proxy = (gpd.read_parquet(ng_storage_comp_station_proxy_path)
                                     .loc[:, ["year", "geometry", "rel_emi"]]
                                     )

    # Read in the ng_trans_comp_station_proxy
    ng_trans_comp_station_proxy = (gpd.read_parquet(ng_trans_comp_station_proxy_path)
                                   .loc[:, ["year", "geometry", "rel_emi"]]
                                   )

    # Calculate the fraction of generator emissions that occur at transmission vs. storage compressor stations
    # Calculate as the average ratio of the horsepower of engines and turbines at transmission stations relative to at storage stations
    # In the GHGI: Horsepower data is calcualted for 1992 based on GRI study. Factors relative to 1992 are applied to 1992 values to complete the timeseries

    # Get column names on Activity Factors tab of the Natural Gas Transmission & Storage GHGI spreadsheet
    names = pd.read_excel(ng_ghgi_path, sheet_name = "Activity Factors", usecols="E:AM", skiprows = 6, header = 0, nrows = 1)
    colnames = names.columns.values

    # Get and clean up generators Activity Factors data
    #   - Engines (Transmission); Turbines (Transmission)
    #   - Engines (Storage); Turbines (Storage)
    EPA_Gen_AF = pd.read_excel(ng_ghgi_path, sheet_name = "Activity Factors", usecols = "E:AM", skiprows = 47, names = colnames, nrows = 4)
    # Get rid of parentheses
    EPA_Gen_AF['Source']= EPA_Gen_AF['Source'].str.replace("(","")
    EPA_Gen_AF['Source']= EPA_Gen_AF['Source'].str.replace(")","")
    # Remove extra spaces in the Source names
    EPA_Gen_AF['Source']= EPA_Gen_AF['Source'].str.replace(r"^ +| +$", r"", regex=True)
    # Drop unneeded columns
    EPA_Gen_AF = EPA_Gen_AF.drop(columns = ['Units'])
    EPA_Gen_AF = EPA_Gen_AF.drop(columns = [*range(1990, min_year,1)])
    EPA_Gen_AF.reset_index(inplace=True, drop=True)

    # Create array to store fraction of generator emissions from transmission stations
    frac_gen_trans = np.zeros(len(years))
    
    eng_trans = EPA_Gen_AF.loc[EPA_Gen_AF['Source'].str.contains('Engines Transmission')].reset_index(drop=True)
    turb_trans = EPA_Gen_AF.loc[EPA_Gen_AF['Source'].str.contains('Turbines Transmission')].reset_index(drop=True)
    eng_stor = EPA_Gen_AF.loc[EPA_Gen_AF['Source'].str.contains('Engines Storage')].reset_index(drop=True)
    turb_stor = EPA_Gen_AF.loc[EPA_Gen_AF['Source'].str.contains('Turbines Storage')].reset_index(drop=True)

    print('Fraction of Generator Emissions from Transmission Stations (relative to Storage Stations):')
    for iyear in np.arange(0, len(years)):
        frac_gen_trans[iyear] = ((eng_trans.iloc[0,iyear+1]/(eng_trans.iloc[0,iyear+1] + eng_stor.iloc[0,iyear+1])) +
                                 (turb_trans.iloc[0,iyear+1]/(turb_trans.iloc[0,iyear+1] + turb_stor.iloc[0,iyear+1]))) / 2
        print('Year', years[iyear], ':', frac_gen_trans[iyear])
    
    # Calculate the rel_emi for the generators_proxy by applying the Fraction of
    # Generator Emissions from Transmission Stations to the rel_emi of the storage and
    # transmission proxies.
    generators_proxy = pd.DataFrame()
    for iyear in np.arange(0, len(years)):
        frac_gen_trans_iyear = frac_gen_trans[iyear]
        storage_proxy_iyear = (ng_storage_comp_station_proxy
                               .query(f"year == {years[iyear]}")
                               .assign(rel_emi=lambda df: df['rel_emi'] * (1 - frac_gen_trans_iyear))
                               )
        trans_proxy_iyear = (ng_trans_comp_station_proxy
                             .query(f"year == {years[iyear]}")
                             .assign(rel_emi=lambda df: df['rel_emi'] * frac_gen_trans_iyear)
                             )
        generators_proxy = pd.concat([generators_proxy, storage_proxy_iyear, trans_proxy_iyear]).reset_index(drop=True)

    # Convert proxy to geodataframe
    proxy_gdf = (
        gpd.GeoDataFrame(generators_proxy)
        .loc[:, ["year", "geometry", "rel_emi"]]
        .astype({"rel_emi":float, "year": int})
    )

    # Check that relative emissions sum to 1.0 each year
    sums = proxy_gdf.groupby(["year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year; {sums}"  # assert that the sums are close to 1

    # Output Proxy Parquet Files
    proxy_gdf.to_parquet(generators_proxy_output_path)

    return None
