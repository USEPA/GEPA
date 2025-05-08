# %% STEP 0. Load packages, configuration files, and local parameters ------------------
# for testing/development
%load_ext autoreload
%autoreload 2
# %%

import logging
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

from gch4i.config import logging_dir, prelim_gridded_dir
from gch4i.gridding_utils import EmiProxyGridder, GriddingInfo, GroupGridder

gpd.options.io_engine = "pyogrio"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("future.no_silent_downcasting", True)
# %%
# create the log file for today that writes out the status of all the gridding
# operations.
logger = logging.getLogger(__name__)
now = datetime.now()
formatted_today = now.strftime("%Y-%m-%d")
formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")
log_file_path = logging_dir / f"gridding_log_{formatted_today}.log"
# start the log file
logging.basicConfig(
    filename=log_file_path,
    encoding="utf-8",
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
# %%
# get the object needed to manage the gridding operations
g_info = GriddingInfo(update_mapping=True, save_file=True)
# display the overall status of emi/proxy pairs
g_info.display_all_pair_statuses()
# %%
# gch4i_name = "3A_enteric_fermentation"
gch4i_name = "3C_rice_cultivation"

gridding_rows = g_info.pairs_ready_for_gridding_df.query(
    f"gch4i_name == '{gch4i_name}'"
)
for emi_proxy_data in tqdm(
    gridding_rows.itertuples(index=False),
    total=len(gridding_rows),
    desc="gridding emi/proxy pairs"
):

    epg = EmiProxyGridder(emi_proxy_data)
    epg.run_gridding()
    print(epg.base_name, epg.status)

g_info.get_ready_groups()
if g_info.group_ready_status.loc[gch4i_name].iloc[0]:
    gridding_group_data = g_info.ready_groups_df.query(f"gch4i_name == '{gch4i_name}'")
    gg = GroupGridder(gch4i_name, gridding_group_data, prelim_gridded_dir)
    gg.run_gridding()
else:
    print("one or more emi/proxy pairs are not ready for gridding.")
# %%
