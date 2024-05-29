import xml.etree
import xml.etree.ElementTree
import pandas as pd

# %%
# get subpart K data: join "k_subpart_level_information" with "pub_dim_facility"
# (needed for lat/lon). Get only the methane records.
subart_k_api_url = (
    "https://data.epa.gov/efservice/k_subpart_level_information/"
    "pub_dim_facility/ghg_name/=/Methane/CSV"
)
subpart_k_df = pd.read_csv(subart_k_api_url)
subpart_k_df["facility_id"].nunique()
subpart_k_df.head()
# %%
# get FRS data from FRS_FACILITY_SITE joined with GEO_FACILITY_POINT tables.
# get only the first 10 rows



frs_naics_url = (
    "https://data.epa.gov/efservice/"
    "FRS_NAICS/"
    f"naics_code/=/{COMPOSTING_FRS_NAICS_CODE}/"
    # "rows/0:100/"
    "JSON"
)
frs_naics_df = pd.read_json(frs_naics_url).set_index(["pgm_sys_acrnm", "pgm_sys_id"])
frs_naics_df.head()
# %%

frs_location_url = (
    "https://data.epa.gov/efservice/FRS_PROGRAM_FACILITY/GEO_FACILITY_POINT/"
    "rows/0:10000/"
    "CSV"
)
frs_location_df = pd.read_csv(frs_location_url)
# %%
frs_w_naics_df = frs_location_df.set_index(["pgm_sys_acrnm", "pgm_sys_id"]).join(
    frs_naics_df.set_index(["pgm_sys_acrnm", "pgm_sys_id"]),
    # how="left",
    rsuffix="_drop",
)
frs_w_naics_df.head()

# %%
frs_w_naics_df["naics_code"].value_counts()
# %%
# https://www.epa.gov/frs/frs-physical-data-model
# describes the database model
# FRS_NAICS --> key 1 (["pgm_sys_acrnm", "pgm_sys_id"]) --> FRS_PROGRAM_FACILITY
# --> 
# %%
import requests
import xml.etree.ElementTree as ET
tmp_url = "https://data.epa.gov/efservice/FRS_PROGRAM_FACILITY/GEO_FACILITY_POINT/COUNT"
with requests.get(tmp_url) as r:
    fac_count = ET.fromstring(r.content)[0][0].text
fac_count
# %%
from itertools import chain
# %%
chain?
# %%
def chunk_in_range(chunk_size, max):
    i = 0
    result_list = []
    while i < max:
        result_list.append(f"{i}:{i+chunk_size - 1}")
        i = i + (chunk_size)
    return result_list
query_chunks = chunk_in_range(10_000, int(fac_count))
# %%
len(query_chunks)
# %%
