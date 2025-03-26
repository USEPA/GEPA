import logging
import pandas as pd
from gch4i.utils import normalize
import geopandas as gpd
import sqlite3

logger = logging.getLogger(__name__)


def update_status(cursor, conn, gch4i_name, emi_id, proxy_id, status):
    cursor.execute(
        """
    INSERT INTO gridding_status (gch4i_name, emi_id, proxy_id, status)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(gch4i_name, emi_id, proxy_id) DO UPDATE SET status=excluded.status
    """,
        (gch4i_name, emi_id, proxy_id, status),
    )
    conn.commit()


def get_status_table(status_db_path, working_dir, the_date):
    # get a numan readable version of the status database
    conn = sqlite3.connect(status_db_path)
    status_df = pd.read_sql_query("SELECT * FROM gridding_status", conn)
    conn.close()
    status_df.to_csv(working_dir / f"gridding_status_{the_date}.csv", index=False)
    return status_df


def get_status(gch4i_name, emi_id, proxy_id):
    cursor.execute(
        """
        SELECT status FROM gridding_status
        WHERE gch4i_name = ? AND emi_id = ? AND proxy_id = ?
        """,
        (gch4i_name, emi_id, proxy_id),
    )
    result = cursor.fetchone()
    return result[0] if result else None


class EmiProxyGridder:

    def __init__(self, row, db_path):
        self.row = row
        self.gch4i_name = row.gch4i_name
        self.emi_id = row.emi_id
        self.proxy_id = row.proxy_id
        self.proxy_time_step = row.proxy_time_step
        self.proxy_time_step = row.proxy_time_step
        self.proxy_has_year_col = row.proxy_has_year_col
        self.proxy_has_month_col = row.proxy_has_month_col
        self.proxy_has_year_month_col = row.proxy_has_year_month_col
        self.proxy_has_rel_emi_col = row.proxy_has_rel_emi_col
        self.proxy_rel_emi_col = row.proxy_rel_emi_col
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.status = get_status(self.gch4i_name, self.emi_id, self.proxy_id)
        if self.status is None:
            self.status = "not started"
            update_status(
                self.cursor,
                self.conn,
                self.gch4i_name,
                self.emi_id,
                self.proxy_id,
                self.status,
            )
