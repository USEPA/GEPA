# %%
import sqlite3
from pathlib import Path
import pandas as pd
from gch4i.config import V4_DATA_PATH, v4_status_db_path, v4_logging_dir

# %%
# Create a connection to the SQLite database



def initiate_status_and_logging_db():
    """Create the gridding_status table if it doesn't exist."""
    conn = sqlite3.connect(v4_status_db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS gridding_status (
        gch4i_name TEXT,
        emi_id TEXT,
        proxy_id TEXT,
        status TEXT,
        PRIMARY KEY (gch4i_name, emi_id, proxy_id)
    )
    """
    )
    conn.commit()
    conn.close()

    data_guide_path = Path(
        V4_DATA_PATH.parents[0] / "gch4i_data_guide_v4.xlsx"
    )
    mapping_df = pd.read_excel(
        data_guide_path, sheet_name="emi_proxy_mapping", engine="openpyxl"
    )
    mapping_df.head()


    for gch4i_name in mapping_df.gch4i_name:
        out_qc_dir = v4_logging_dir / gch4i_name
        out_qc_dir.mkdir(exist_ok=True, parents=True)
        # all_group_files = list(v4_logging_dir.rglob(f"{gch4i_name}*"))
        # all_group_files = [x for x in all_group_files if not x.is_dir()]
        # all_group_files = [x for x in all_group_files if x.parent != out_qc_dir]
        # if all_group_files:
        #     print(f"Moving {len(all_group_files)} files to {out_qc_dir}")
        #     for file in all_group_files:
        #         destination = out_qc_dir / file.name
        #         if destination.exists():
        #             destination.unlink()
        #         file.rename(destination)

initiate_status_and_logging_db()
# %%
# ======================================================================================
# prepare and clean up the logging and QC output directory
# ======================================================================================


# %%
def delete_db_entry(gch4i_name, emi_id, proxy_id):
    """Delete a specific entry from the gridding_status table."""
    cursor.execute(
        "DELETE FROM gridding_status WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?",
        (gch4i_name, emi_id, proxy_id),
    )
    conn.commit()
# %%

# delete_db_entry("1B2biv_ng_transmission_storage", "storage_wells_blowout_emi", "TBD - HANNAH TO FILL")
# # %%
