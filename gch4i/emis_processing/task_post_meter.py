from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task, mark

from gch4i.config import (emi_data_dir_path, ghgi_data_dir_path, max_year,
                          min_year)
from gch4i.utils import tg_to_kt, us_state_to_abbrev

post_meter_dir_path = ghgi_data_dir_path / "post_meter"


@mark.persist
@task(id="post_meter_emi")
def task_get_stationary_combustion_inv_data(
    #INPUT
    comm_customers_input_path: Path = post_meter_dir_path / "Emi_CommCustomers.xlsx",
    ind_egu_input_path: Path = post_meter_dir_path / "Emi_IndEGU.xlsx",
    cng_vehicles_input_path: Path = post_meter_dir_path / "Emi_CNGVehicles.xlsx",
    res_customers_input_path: Path = post_meter_dir_path / "Emi_ResCustomers.xlsx",

    #OUTPUT 
    postmeter_commercial_emi_output_path: Path = emi_data_dir_path / "postmeter_commercial_emi.csv",
    postmeter_egus_emi_output_path: Path = emi_data_dir_path / "postmeter_egus_emi.csv",
    postmeter_ng_vehicles_emi_output_path: Path = emi_data_dir_path / "postmeter_ng_vehicles_emi.csv",
    postmeter_residential_emi_output_path: Path = emi_data_dir_path / "postmeter_residential_emi.csv",
) -> None:

    ########################
    # Read & clean inputs
    ########################
    
    # commercial customers
    comm_customers_df = (
        pd.read_excel(
            comm_customers_input_path,
            sheet_name="Sheet1",
            skiprows=0,
            nrows=52,
            usecols="A:AH",
        ).melt(
            id_vars=['State',],
            var_name='Year',
            value_name='ghgi_ch4_kt',
        ).query("Year <= @max_year and Year >= @min_year"
        ).rename(columns={"State": "state_code", "Year": "year"}
        ).map(us_state_to_abbrev
        ).sort_values(by=['year', 'state_code']
        )
    )

    # industrial egus
    ind_egu_df = (
        pd.read_excel(
            ind_egu_input_path,
            sheet_name="Sheet1",
            skiprows=0,
            nrows=52,
            usecols="A:AH",
        ).melt(
            id_vars=['State',],
            var_name='Year',
            value_name='ghgi_ch4_kt',
        ).query("Year <= @max_year and Year >= @min_year"
        ).rename(columns={"State": "state_code", "Year": "year"}
        ).map(us_state_to_abbrev
        ).sort_values(by=['year', 'state_code']
        )
    )

    # cng vehicles
    cng_vehicles_df = (
        pd.read_excel(
            cng_vehicles_input_path,
            sheet_name="Sheet1",
            skiprows=0,
            nrows=52,
            usecols="A:AH",
        ).melt(
            id_vars=['State',],
            var_name='Year',
            value_name='ghgi_ch4_kt',
        ).query("Year <= @max_year and Year >= @min_year"
        ).rename(columns={"State": "state_code", "Year": "year"}
        ).map(us_state_to_abbrev
        ).sort_values(by=['year', 'state_code']
        )
    )

    # residential customers
    res_customers_df = (
        pd.read_excel(
            res_customers_input_path,
            sheet_name="Sheet1",
            skiprows=0,
            nrows=52,
            usecols="A:AH",
        ).melt(
            id_vars=['State',],
            var_name='Year',
            value_name='ghgi_ch4_kt',
        ).query("Year <= @max_year and Year >= @min_year"
        ).rename(columns={"State": "state_code", "Year": "year"}
        ).map(us_state_to_abbrev
        ).sort_values(by=['year', 'state_code']
        )
    )


    ########################
    # Create outputs
    ########################

    # commercial customers
    comm_customers_df.to_csv(postmeter_commercial_emi_output_path, index=False)

    # industrial egus
    ind_egu_df.to_csv(postmeter_egus_emi_output_path, index=False)

    # cng vehicles
    cng_vehicles_df.to_csv(postmeter_ng_vehicles_emi_output_path, index=False)

    # residential customers
    res_customers_df.to_csv(postmeter_residential_emi_output_path, index=False)

