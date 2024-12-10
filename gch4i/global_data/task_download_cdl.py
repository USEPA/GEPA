from pathlib import Path
from typing import Annotated

import requests
from pytask import Product, mark, task

from gch4i.config import sector_data_dir_path, years

nass_cdl_path = sector_data_dir_path / "nass_cdl"

_ID_TO_KWARGS_DL_CDL = {}
for year in years:
    file_name = f"{year}_30m_cdls.zip"
    url = (
        "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/"
        f"datasets/{file_name}"
    )

    output_path = nass_cdl_path / file_name
    _ID_TO_KWARGS_DL_CDL[f"nass_cdl_{year}"] = dict(
        url=url,
        output_path=output_path,
    )
_ID_TO_KWARGS_DL_CDL

for _id, kwargs in _ID_TO_KWARGS_DL_CDL.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_cdl(url: str, output_path: Annotated[Path, Product]) -> None:
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("File downloaded successfully!")
        except requests.exceptions.RequestException as e:
            print("Error downloading the file:", e)
# %%
import pytask

session = pytask.build(tasks=[task_download_cdl(**kwargs) for _, kwargs in _ID_TO_KWARGS_DL_CDL.items()])

# %%
