import pandas as pd
from os import path
import numpy as np
from glob import glob

dbdir = (
    r"C:\Users\Takeshi Ikuma\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database"
)

_files = [
    glob(path.join(dbdir, vtype, atype, "*.NSP"))
    for vtype in ("NORM", "PATHOL")
    for atype in ("AH", "RAINBOW")
]
ah_files = [*_files[0], *_files[2]]
ra_files = [*_files[1], *_files[3]]

df = pd.read_table(
    path.join(dbdir, "EXCEL50", "TEXT", "KAYCDALL.TXT"),
    skipinitialspace=True,
    dtype={
        # fmt: off
        **{col: 'Int32' for col in ["AGE", "#", "NVB", "SH", "NUV", "SEG", "PER"]},
        **{col: "string" for col in ["PAT_ID", "FILE VOWEL 'AH'"]},
        **{col: "category" for col in ["SEX", "DIAGNOSIS", "LOCATION",  "SMOKE", "NATLANG", "ORIGIN"]},
        **{col: float for col in ["Fo", "To", "Fhi", "Flo", "STD", "PFR", "Fftr", "Fatr", "Tsam", "Jita", 
                                  "Jitt", "RAP", "PQ", "sPPQ", "vFo", "ShdB", "Shim", "APQ", "sAPQ", 
                                  "vAm", "NHR", "VTI", "SPI", "FTRI", "ATRI", "DVB", "DSH", "DUV"]}
        # fmt: on
    },
    parse_dates=["VISITDATE"],
)

# add PAT_ID if missing (use unique ID based on the file name)
tf = pd.isna(df["PAT_ID"])
for row in np.where(tf)[0]:
    id = df.at[row, "FILE VOWEL 'AH'"][:3]
    ids = df["PAT_ID"][df["PAT_ID"].str.startswith(id, na=False)].tolist()
    if len(ids):
        for i in range(1000):
            id_ = f"{id}{i:03}"
            if id_ not in ids:
                break
    else:
        id_ = f"{id}000"
    df.at[row, "PAT_ID"] = id_

# assign for rainbow
ra_name = [path.basename(f)[:3] for f in ra_files]
ra_day = [int(path.basename(f)[3:5]) for f in ra_files]

# remove entries without PAT_ID or FILE VOWEL 'AH'
tf = pd.isna(df["FILE VOWEL 'AH'"])
for row in np.where(tf)[0]:
    df.at[row, "PAT_ID"]
    df.at[row, "VISITDATE"].dt.day
# ids = df[miss]
# print(ids)

# separate diagnosis
# df_dx = df[]


# miss = pd.isna(df[["PAT_ID", "FILE VOWEL 'AH'"]])
# print(np.sum(miss, 0), np.sum(np.any(miss, 1), 0))

# print(df.info())

# print(dict(df.dtypes))

# drop
# drop_duplicates
