"""
I am looking at the different consolidated facility files that the EPA offers.

I've downloaded the full FRS dataset from this link:
https://ordsext.epa.gov/FLA/www3/state_files/national_combined.zip
"""

# %%
from pathlib import Path
import pandas as pd
import duckdb

# %%
frs_path = Path(
    (
        "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/"
        "Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/raw/"
        "NATIONAL_FACILITY_FILE.CSV"
    )
)

fac_df = duckdb.execute(
    (
        # "SELECT primary_name, state_code, latitude83, longitude83 "
        "SELECT * "
        f"FROM '{frs_path}' "
        "LIMIT 100"
    )
).df()
fac_df.head()

# %%
