# %%
from pathlib import Path
import duckdb
# %%
frs_path = Path("C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/raw/NATIONAL_FACILITY_FILE.CSV")
# %%
fac_df = duckdb.execute(
    (
        "SELECT primary_name, state_code, latitude83, longitude83 "
        f"FROM '{frs_path}' "
        # "WHERE state_code == 'NC' "
        # "AND primary_name LIKE '%CARGILL%' "
        "WHERE primary_name LIKE '%CARBIDE%' "
    )
).df()
fac_df.head()

# %%
fac_df.shape
# %%
frs_cols = duckdb.execute(
    (
        "SELECT * "
        f"FROM '{frs_path}' "
        "LIMIT 5"
    )
).df()
frs_cols.columns
# %%
frs_cols
# %%

frs_cols["LOCATION_DESCRIPTION"]
# %%
