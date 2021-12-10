import pandas as pd
from os import path
import numpy as np
from glob import glob
import re

dbdir = (
    r"C:\Users\tikuma\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database"
    # r"C:\Users\Takeshi Ikuma\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database"
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
    # id = df.at[row, "FILE VOWEL 'AH'"][:3]
    # ids = df["PAT_ID"][df["PAT_ID"].str.startswith(id, na=False)].tolist()
    # if len(ids):
    #     for i in range(1000):
    #         id_ = f"{id}{i:03}"
    #         if id_ not in ids:
    #             break
    # else:
    #     id_ = f"{id}000"
    # df.at[row, "PAT_ID"] = id_
    df.at[row, "PAT_ID"] = df.at[row, "FILE VOWEL 'AH'"]

# split dx
df_dx = df[["PAT_ID", "VISITDATE", "DIAGNOSIS"]]
df.drop(columns=["#", "DIAGNOSIS"], inplace=True)
df.drop_duplicates(subset=["PAT_ID", "VISITDATE"], inplace=True)
df.reset_index(inplace=True)

# assign for rainbow
ra_names = set((path.basename(f) for f in ra_files))

# remove entries without PAT_ID or FILE VOWEL 'AH'
n = df.shape[0]
rnames = [None]*n
isnorms = np.empty(n, bool)
tf = np.zeros(n, bool)
for row in range(n):
    aname = df.at[row, "FILE VOWEL 'AH'"]
    if pd.isna(aname):
        id = df.at[row, "PAT_ID"][:3]
        rname = f"{id}1NRL.NSP"
        isnorms[row] = isnorm = rname in ra_names
        if not isnorm:
            day = df.at[row, "VISITDATE"].day
            rname = f"{id}{day:02}R.NSP"
    else:
        isnorms[row] = isnorm = aname.endswith("NAL.NSP")
        rname = (
            re.sub(r"NAL\.NSP$", "NRL.NSP", aname)
            if isnorm
            else re.sub(r"AN\.NSP$", "R.NSP", aname)
        )

    if rname in ra_names:
        rnames[row] = rname
    else:
        tf[row] = True

print(rnames)

df.insert(3, "FILE RAINBOW", rnames)
df.insert(3, "NORM", isnorms)
df.drop(index=np.where(tf)[0], inplace=True)
df.reset_index(inplace=True)

print(df)