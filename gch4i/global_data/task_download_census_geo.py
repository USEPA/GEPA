# %%
# download the census geometry files we need for gridded methane v3.
from pathlib import Path
from typing import Annotated

import requests
from pytask import Product, mark, task

from gch4i.config import census_reference_geometry_list, global_data_dir_path


def create_kwargs(geometry_list):
    id_to_kwargs = {}
    # for each of the geometry types listed
    for geometry_type in geometry_list:
        # create the URL
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER2020/{geometry_type.upper()}/"
            f"tl_2020_us_{geometry_type.lower()}.zip"
        )
        # the output file is the same file name as TIGER filename
        output_path = global_data_dir_path / url.split("/")[-1]
        id_to_kwargs[geometry_type] = {"url": url, "output_path": output_path}
    return id_to_kwargs


_ID_TO_KWARGS = create_kwargs(census_reference_geometry_list)

for _id, kwargs in _ID_TO_KWARGS.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_census_geos(
        url: str, output_path: Annotated[Path, Product]
    ) -> None:
        with output_path.open("wb") as dst:
            with requests.get(url, stream=True) as r:
                dst.write(r.content)


# %%
