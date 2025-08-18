# %%
import sqlite3
from pathlib import Path
from gch4i.config import status_db_path, logging_dir

# %%
# Create a connection to the SQLite database

conn = sqlite3.connect(status_db_path)
cursor = conn.cursor()

# Create a table to store the status of each row
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
# %%
# ======================================================================================
# prepare and clean up the logging and QC output directory
# ======================================================================================
for gch4i_name in mapping_df.gch4i_name:
    all_group_files = list(logging_dir.rglob(f"{gch4i_name}*"))
    out_qc_dir = logging_dir / gch4i_name
    out_qc_dir.mkdir(exist_ok=True, parents=True)
    all_group_files = [x for x in all_group_files if not x.is_dir()]
    all_group_files = [x for x in all_group_files if x.parent != out_qc_dir]
    if all_group_files:
        print(f"Moving {len(all_group_files)} files to {out_qc_dir}")
        for file in all_group_files:
            destination = out_qc_dir / file.name
            if destination.exists():
                destination.unlink()
            file.rename(destination)

# %%
def delete_db_entry(gch4i_name, emi_id, proxy_id):
    """Delete a specific entry from the gridding_status table."""
    cursor.execute(
        "DELETE FROM gridding_status WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?",
        (gch4i_name, emi_id, proxy_id),
    )
    conn.commit()
# %%

delete_db_entry("1B2biv_ng_transmission_storage", "storage_wells_blowout_emi", "TBD - HANNAH TO FILL")
# %%
