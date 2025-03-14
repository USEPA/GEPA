# %%
from datetime import datetime
from numpy import nan
import pandas as pd
from gch4i.config import (
    emi_data_dir_path,
    proxy_data_dir_path,
    V3_DATA_PATH,
    tmp_data_dir_path,
)


final_gridded_data_dir = V3_DATA_PATH / "final_gridded_methane"

work_dir = V3_DATA_PATH.parent.absolute()
guide = work_dir.parent.absolute() / "gch4i_data_guide_v3.xlsx"
guide_sheet = pd.read_excel(guide, sheet_name="emi_proxy_mapping")
prog_file = (
    work_dir
    / "v3_progress"
    / f"gch4i_progress_{datetime.now().date().strftime('%Y%m%d')}.xlsx"
)
emi_files = [f for f in list(emi_data_dir_path.glob("*.*"))]
emi_labels = [f.stem for f in emi_files]
proxy_files = [f for f in list(proxy_data_dir_path.glob("*.*"))]
proxy_labels = [f.stem for f in proxy_files]
gch4i_rows = []
missing_emi_rows = []
missing_proxy_rows = []
missing_final_data_rows = []
groups = guide_sheet.gch4i_name.unique()
# %%

def main():
    # %%
    for (
        g
    ) in groups:  # loop through gch4i_names to check for input, emi, and proxy files
        df = guide_sheet[guide_sheet.gch4i_name == g]
        g_emis = df.emi_id.unique()
        g_proxies = []
        for p in df.proxy_id.unique():  # loop through proxy value to check for multiple
            if ";" in str(p):
                g_proxies.extend(p.split("; "))
            else:
                g_proxies.append(p)
        # check for empty values in the input file column
        empty = [nan, "", " ", None, 0]
        has_emi_inputs = not any(v in empty for v in df.file_name.unique())
        emis_done = set(g_emis) <= set(emi_labels)  # all emis present as files
        proxies_done = set(g_proxies) <= set(
            proxy_labels
        )  # all proxies present as files
        # create dictionary/row for gch4i_name and add to list of gch4i rows

        prelim_path = tmp_data_dir_path / f"{g}_ch4_emi_flux.nc"
        prelim_data_done = prelim_path.exists()
        final_path = final_gridded_data_dir / f"{g}_ch4_emi_flux.nc"
        final_data_done = final_path.exists()

        gch4i_row = {
            "gch4i_name": g,
            "has_emi_inputs": has_emi_inputs,
            "all_emis_done": emis_done,
            "all_proxies_done": proxies_done,
            "prelim_data_done": prelim_data_done,
            "final_data_done": final_data_done,
        }
        gch4i_rows.append(gch4i_row)
        if not emis_done:  # create dict of missing emi files if any are missing
            g_emis = ["[blank emi]" if e in empty else e for e in g_emis]
            missing_emis = [
                {"gch4i_name": g, "emi_not_done": e}
                for e in g_emis
                if e not in emi_labels
            ]
            missing_emi_rows.extend(missing_emis)
        if not proxies_done:  # create dict of missing proxy files if any are missing
            missing_proxies = [
                {"gch4i_name": g, "proxy_not_done": p}
                for p in g_proxies
                if p not in proxy_labels
            ]
            missing_proxy_rows.extend(missing_proxies)

    with pd.ExcelWriter(prog_file, engine="xlsxwriter") as writer:
        # write gch4i_name emi/proxy progress to the first sheet
        pd.DataFrame.from_dict(gch4i_rows).to_excel(
            writer, index=False, sheet_name="progress"
        )
        workbook = writer.book
        prog_sheet = writer.sheets["progress"]
        gch4i_len = len(gch4i_rows) + 1
        format0 = workbook.add_format({"border": 1})
        format1 = workbook.add_format(
            {"bg_color": "#FFC7CE", "font_color": "#9C0006", "border": 1}
        )
        format2 = workbook.add_format(
            {"bg_color": "#C6EFCE", "font_color": "#006100", "border": 1}
        )
        prog_sheet.conditional_format(
            f"$A$2:$A${gch4i_len}", {"type": "no_blanks", "format": format0}
        )
        prog_sheet.conditional_format(
            f"$B$2:$F${gch4i_len}",
            {"type": "cell", "criteria": "=", "value": False, "format": format1},
        )
        prog_sheet.conditional_format(
            f"$B$2:$F${gch4i_len}",
            {"type": "cell", "criteria": "=", "value": True, "format": format2},
        )
        prog_sheet.autofit()
        if missing_emi_rows:
            # write missing emi_file dict to the second sheet
            pd.DataFrame.from_dict(missing_emi_rows).to_excel(
                writer, index=False, sheet_name="missing_emis"
            )
            emi_sheet = writer.sheets["missing_emis"]
            emi_sheet.conditional_format(
                f"$A$2:$B${len(missing_emi_rows) + 1}",
                {"type": "no_blanks", "format": format0},
            )
            emi_sheet.autofit()
        # write missing proxy file dict to the third sheet
        if missing_proxy_rows:
            (
                pd.DataFrame.from_dict(missing_proxy_rows)
                .groupby("proxy_not_done")["gch4i_name"]
                .unique()
                .apply(lambda x: "; ".join(x))
                .reset_index()
                .to_excel(writer, index=False, sheet_name="missing_proxies")
            )
            proxy_sheet = writer.sheets["missing_proxies"]
            proxy_sheet.conditional_format(
                f"$A$2:$B${len(missing_proxy_rows) + 1}",
                {"type": "no_blanks", "format": format0},
            )
            proxy_sheet.autofit()

# %%
if __name__ == "__main__":
    main()

# %%
