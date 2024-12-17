"""
Name:                   task_stationary_combustion.py
Date Last Modified:     2024-07-06
Authors Name:           John Bollenbacher (RTI International)
Purpose:                Mapping of stationary combustion emissions to State, Year, emissions format
Input Files:            - Stationary non-CO2 InvDB State Breakout_2021.xlsx
Output Files:           - Do not use output. This is a single function script.
Notes:                  
"""
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task, mark

from gch4i.config import (emi_data_dir_path, ghgi_data_dir_path, max_year,
                          min_year)
from gch4i.utils import tg_to_kt


@mark.persist
@task(id="ab_coal_emi")
def task_get_stationary_combustion_inv_data(
    #INPUT
    stationary_combustion_input_path: Path = ghgi_data_dir_path
    / "Stationary non-CO2 InvDB State Breakout_2021.xlsx",

    #OUTPUT commercial
    stat_comb_comm_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_comm_emi.csv",

    #OUTPUT industrial
    stat_comb_ind_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_indu_emi.csv",

    #OUTPUT residential
    stat_comb_res_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_res_emi.csv",

    #OUTPUT us_territories
    stat_comb_us_territories_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_us_territories_emi.csv",

    #OUTPUT electric, by fuel type
    stat_comb_elec_coal_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_elec_coal_emi.csv",
    stat_comb_elec_gas_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_elec_gas_emi.csv",
    stat_comb_elec_oil_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_elec_oil_emi.csv",
    stat_comb_elec_wood_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "stat_comb_elec_wood_emi.csv",
) -> None:

    ########################
    # Read & clean inputs
    ########################
    stationary_combustion_df = (
        pd.read_excel(
            stationary_combustion_input_path,
            sheet_name="InvDB",
            skiprows=15,
            usecols="A:AO",
        ).filter((['GHG', 'Subsource', 'Fuel', 'State']+[y for y in range(min_year, max_year+1)])
        ).query(
        "GHG == 'CH4'"# and Category in ['Commercial', 'Electricity Generation']"
        ).drop(columns = ['GHG']
        ).rename(columns={'Subsource':'Category'})
        ).melt(
            id_vars=['Category', 'Fuel', 'State'],
            var_name='Year',
            value_name='CH4_emissions_Tg',
        ).query("Year <= @max_year and Year >= @min_year"
        ).replace(to_replace = { #map fuel types to standard names
                'Coal': 'coal',
                'Natural Gas': 'gas',
                'Petroleum': 'oil',
                'Biomass': 'wood', #we confirmed that all biomass in this dataset is wood
        }
    )
    stationary_combustion_df['ghgi_ch4_kt'] = (stationary_combustion_df['CH4_emissions_Tg']
                                                    ).apply(pd.to_numeric, errors="coerce"
                                                    ).apply(lambda x: x*tg_to_kt)
    stationary_combustion_df = stationary_combustion_df.drop(columns=['CH4_emissions_Tg'], 
                                                      ).rename(columns={'State':'state_code',
                                                                        'Year':'year',}
                                                      ).dropna(subset=['ghgi_ch4_kt']) #drop rows with missing values, per Nick K. instruction


    ########################
    # create outputs
    ########################
    # create output files with columns ['State', 'Year', 'ghgi_ch4_kt'], per category
    # commercial
    stationary_combustion_df.query("Category == 'Commercial'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_comm_emi_output_path, index=False)
    
    # industrial
    stationary_combustion_df.query("Category == 'Industrial'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_ind_emi_output_path, index=False)
    
    # residential
    stationary_combustion_df.query("Category == 'Residential'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_res_emi_output_path, index=False)

    # us_territories
    stationary_combustion_df.query("Category == 'US Territories'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_us_territories_emi_output_path, index=False)
    #electrical generation disaggregated by fuel type
    #coal
    stationary_combustion_df.query("Category == 'Electricity Generation' and Fuel == 'coal'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_elec_coal_emi_output_path, index=False)
    #gas
    stationary_combustion_df.query("Category == 'Electricity Generation' and Fuel == 'gas'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_elec_gas_emi_output_path, index=False)
    #oil
    stationary_combustion_df.query("Category == 'Electricity Generation' and Fuel == 'oil'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_elec_oil_emi_output_path, index=False)
    #wood
    stationary_combustion_df.query("Category == 'Electricity Generation' and Fuel == 'wood'"
                               ).filter(['year','state_code','ghgi_ch4_kt']
                               ).to_csv(stat_comb_elec_wood_emi_output_path, index=False)
