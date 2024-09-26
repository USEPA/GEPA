import pandas as pd
from gch4i.config import (V3_DATA_PATH)
import os
import shutil

work_dir = V3_DATA_PATH.parent.absolute()
guide = work_dir.parent.absolute() / "gch4i_data_guide_v3.xlsx"
guide_sheet = pd.read_excel(guide, sheet_name="emi_proxy_mapping")
all_ghgi_files = []
# gather list of all files within ghgi dir and its subdirs
for subdir, dirs, files in os.walk(V3_DATA_PATH / "ghgi"):
    for file in files:
        all_ghgi_files.append(subdir + os.sep + file)
groups = guide_sheet.gch4i_name.unique()
for g in groups:  # loop through gch4i_names and ensure each one has a namesake dir
    ghgi_dir = V3_DATA_PATH / "ghgi" / g
    if not os.path.isdir(ghgi_dir):
        print(f"Creating directory: {ghgi_dir}")
        os.mkdir(ghgi_dir)
    df = guide_sheet[guide_sheet.gch4i_name == g]
    for f in df.file_name.unique():  # ensure emi_inputs exist in gch4i folder
        if isinstance(f, str):
            # account for filenames with no extension. Assumes all xlsx for now.
            if ".xlsx" not in f and "." not in f:
                f += ".xlsx"
            if "," in str(f):  # account for comma-separated lists of emi_inputs
                for x in f.split(","):
                    if not os.path.isfile(ghgi_dir / x):
                        print(f"{x} not found for {g}")
                        for af in all_ghgi_files:
                            if os.path.basename(af) == x:
                                shutil.copy2(af, ghgi_dir / x)
                                break
            else:
                if not os.path.isfile(ghgi_dir / f):
                    print(f"{f} not found for {g}")
                    for af in all_ghgi_files:
                        if os.path.basename(af) == f:
                            shutil.copy2(af, ghgi_dir / f)
                            break
