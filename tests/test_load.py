import kpdvdb
import pandas as pd
from os import path
import numpy as np
from glob import glob
import re

dbdir = (
    r"C:\Users\tikum\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database"  # excalibur
    # r"C:\Users\tikuma\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database" # samsung
    # r"C:\Users\Takeshi Ikuma\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database" # olol
)


def test_basics():
    kpdvdb.load_db(dbdir)
    print(kpdvdb.get_sexes())
    print(kpdvdb.get_locations())
    print(kpdvdb.get_natlangs())
    print(kpdvdb.get_origins())
    print(kpdvdb.get_diagnoses())


kpdvdb.load_db(dbdir)

# df = kpdvdb.query()
# df = kpdvdb.query(["MDVP"])
# df = kpdvdb.query(["DIAGNOSES"])

# df = kpdvdb.query(
#     ["NORM", "SEX", "Fo", "NATLANG", "VISITDATE"],
#     NORM=False,
#     SEX="M",
#     Fo=[50, 150],
#     NATLANG=["English", "Japanese"],
#     VISITDATE=[np.datetime64('1993-01-01'),np.datetime64('1994-01-01')],
# )

df = kpdvdb.query(["DIAGNOSES"], DIAGNOSES=["normal voice", "paralysis"])

print(df)
# ["diagnoses"]
# print(df)
