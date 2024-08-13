# %% STEP 0. Load packages, configuration files, and local parameters ------------------
import pandas as pd
from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import tg_to_kt  # Determine if this conversion rate is correct

# %% Grab data
inventory_workbook_path = ghgi_data_dir_path / "Stationary Calcs 90-22_3_12_24_FR.xlsx"

emi_df = pd.read_excel(
            inventory_workbook_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=41,
            usecols="A:BA",
            index_col=None
        )
# %% Melt and form data

emi_df = emi_df.rename(columns=lambda x: str(x).lower()) \
               .rename(columns={'georef': 'state_code'}) \
               .query('ghg == "CH4"') \
               .filter(regex='category|fuel1|19|20') \
               .drop(columns=['subcategory1', 'subcategory2', 'subcategory3', 'subcategory4', 'subcategory5']) 

emi_df.iloc[:, 2:] = emi_df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")

emi_df = emi_df.melt(id_vars=["category", "fuel1"], var_name="year", value_name="ch4_tg")

# %% Read in Acid Rain

acid_workbook_path = ghgi_data_dir_path / "Acid Rain Prog Non CO2 Estimates 2_27_2024.xlsx"

acid_df = pd.read_excel(
            acid_workbook_path,
            sheet_name="ARP 2021",
            # skiprows=15,
            # nrows=41,
            # usecols="A:BA",
            index_col=None
        )

acid_df = acid_df.rename(columns=lambda x: str(x).lower()) \
                 .iloc[:, [0,14,17]] # year is 4

acid_df = acid_df.groupby(["state", "fuel type"]).sum().reset_index()

# %% Calculate category/fuel % by state

# Each state can have 20 emissions options per year
  # 5 categories * 4 fuel types = 20

# For each fuel type (4), calculate the percentage for each category, per year.
  # Use ratios (%) to assign...Think about this more