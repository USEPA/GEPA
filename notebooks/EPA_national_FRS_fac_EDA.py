# %%
from pathlib import Path
import duckdb
# %%
input_path = Path("C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/raw/NATIONAL_FACILITY_FILE.CSV")
# %%
fac_df = duckdb.execute(
    (
        "SELECT primary_name, state_code, latitude83, longitude83 "
        f"FROM '{input_path}' "
        # "WHERE state_code == 'NC'"
        "WHERE primary_name"
    )
).df()
# %%
fac_df.head()
# %%
fac_df.shape
# %%
