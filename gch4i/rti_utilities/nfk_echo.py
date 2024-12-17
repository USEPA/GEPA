# %%
from pathlib import Path
import duckdb

# %%
# https://echo.epa.gov/tools/data-downloads#FRS
echo_path = Path("C:/Users/nkruskamp/Downloads/ECHO_EXPORTER.csv")

# just preview the first 10 rows and look at column names
echo_df = duckdb.execute(f"SELECT * from read_csv('{echo_path}') LIMIT 10").df()
echo_df.columns.tolist()
# %%
all_fac_df = duckdb.execute(
    f"""
    SELECT
        REGISTRY_ID, FAC_STATE, FAC_NAME, FAC_SIC_CODES
    FROM
        read_csv('{echo_path}')
    WHERE
        FAC_ACTIVE_FLAG == 'Y'
"""
).df()
# %%
sic_codes = all_fac_df["FAC_SIC_CODES"].str.split(" ", expand=True)
pp_rows = sic_codes.apply(lambda x: x.str.startswith("26")).any(axis=1)
pp_rows.sum()
# %%
pp_df = all_fac_df.loc[pp_rows]
pp_df.shape
# %%
pp_df["FAC_STATE"].value_counts()
# %%
d_path = Path("C:/Users/nkruskamp/Downloads/NPDES_DMRS_FY2009.csv")
d_df = duckdb.execute(f"SELECT PARAMETER_DESC, DMR_UNIT_DESC, DRM_ from read_csv('{d_path}')").df()
# %%
d_df.head()
# %%
d_df["PARAMETER_DESC"].unique().tolist()
# %%
d_df["DMR_UNIT_DESC"].sort_values().unique().tolist()
# %%
d_df.columns.tolist()
# %%
import pandas as pd
pd.set_option("display.max_columns", None)
# %%
