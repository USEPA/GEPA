from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    emi_data_dir_path,
    ghgi_data_dir_path,
)