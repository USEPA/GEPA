# %%
from gch4i.config import V3_DATA_PATH
import pandas as pd
# %%
in_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# %%
info_df = pd.read_excel(in_path, sheet_name="emi_proxy_mapping")
info_df
# %%
# ======================================================================================
# get the group for each proxy. See which ones have duplicate groups
# ======================================================================================
info_df.groupby("proxy_id")["gch4i_name"].nunique().sort_values().reset_index().to_clipboard()
# %%
n_unique_proxies = info_df.proxy_id.nunique()
n_unique_proxies
# %%
n_unique_emis = info_df.emi_id.nunique()
n_unique_emis
# %%
info_df[["file_name"]].drop_duplicates().reset_index(drop=True).to_clipboard()
# %%
inventory_data = info_df[["gch4i_name", "proxy_id"]].drop_duplicates().to_clipboard()

# %%
