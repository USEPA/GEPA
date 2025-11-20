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

    data_guide_path = Path(V4_DATA_PATH.parents[0] / "gch4i_data_guide_v4.xlsx")
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


# initiate_status_and_logging_db()
# %%
# ======================================================================================
# prepare and clean up the logging and QC output directory
# ======================================================================================


# %%
def delete_entry_in_db(gch4i_name, emi_id, proxy_id):
    """Delete a specific entry from the gridding_status table."""
    conn = sqlite3.connect(v4_status_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM gridding_status WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?",
        (gch4i_name, emi_id, proxy_id),
    )
    conn.commit()
    conn.close()


# %%


# delete_db_entry("1B2biv_ng_transmission_storage", "storage_wells_blowout_emi", "TBD - HANNAH TO FILL")
# # %%
def reset_status_in_db(gch4i_name, emi_id, proxy_id, new_status="not started"):
    """Delete a specific entry from the gridding_status table."""

    conn = sqlite3.connect(v4_status_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE gridding_status SET status = ? WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?",
        (new_status, gch4i_name, emi_id, proxy_id),
    )
    conn.commit()
    conn.close()


# %%
def read_status_table():
    """Read the gridding_status table into a pandas DataFrame."""
    conn = sqlite3.connect(v4_status_db_path)
    df = pd.read_sql_query("SELECT * FROM gridding_status", conn)
    conn.close()
    return df


status_df = read_status_table()
status_df
# %%
reset_path = Path(
    "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/gch4i_v4/oil_and_gas_emi_proxy_pairs_to_grid.xlsx"
)
# %%
reset_df = pd.read_excel(reset_path, engine="openpyxl")
reset_df
# %%
for _, row in reset_df.iterrows():
    print(f"Resetting {row.gch4i_name}, {row.emi_id}, {row.proxy_id}")
    reset_status_in_db(row.gch4i_name, row.emi_id, row.proxy_id)
# %%
