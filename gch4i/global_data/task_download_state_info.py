# %%
from pytask import task, mark, Product
from gch4i.config import global_data_dir_path
from typing import Annotated
from pathlib import Path
import pandas as pd

# %%
@mark.persist
@task(id="download state info")
def task_download_state_info(
    url: str = "https://www2.census.gov/geo/docs/reference/codes2020/national_state2020.txt",
    output_path: Annotated[Path, Product] = global_data_dir_path
    / "national_state2020.csv",
):
    df = pd.read_csv(url, sep="|").rename(columns=str.lower).rename(columns={"state": "state_code"})
    df.to_csv(output_path, index=False)
