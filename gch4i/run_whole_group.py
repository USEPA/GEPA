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

from gch4i.config import logging_dir
from gch4i.gridding_utils import GriddingInfo, run_whole_group

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
run_these_groups = [
    # "1A_mobile_combustion",
    # "1B2biv_ng_transmission_storage",
    # "3A_enteric_fermentation",
    "4A1_4A2_Forest_land_remaining_forest_land",
    "4C1_4C2_Grassland_remaining_grassland",
    # "1B2biv_ng_transmission_storage",
    # "1B2bii_ng_production",
    # "1B2ai_petroleum_exploration",
    # "1B2aii_petroleum_production",
    # "1B2aiii_petroleum_transport",
]

# %%
for gch4i_name in run_these_groups:
    print(f"Running gridding for group: {gch4i_name}")
    run_whole_group(gch4i_name, g_info)

# %%
