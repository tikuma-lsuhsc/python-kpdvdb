from kpdvdb import KPDVDB
import pandas as pd
from os import path
import numpy as np
from glob import glob
import re
import pytest


def load_db(remove_unknowns=False):
    return KPDVDB(
        # r"C:\Users\tikum\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database",  # excalibur
        # r"C:\Users\tikuma\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database", # samsung
        r"C:\Users\Takeshi Ikuma\OneDrive - LSUHSC\data\KayPENTAX Disordered Voice Database",  # olol
        remove_unknowns,
    )


@pytest.fixture(scope="module")
def kpdvdb():
    return load_db()


def test_basics(kpdvdb):
    print(kpdvdb.get_fields())
    print(kpdvdb.get_sexes())
    print(kpdvdb.get_natlangs())
    print(kpdvdb.get_origins())
    print(kpdvdb.get_diagnoses())
    print(kpdvdb.get_diagnoses(True))


def test_only_knowns():
    db = load_db(True)
    assert not db._df["VISITDATE"].isna().sum()


def test_query(kpdvdb):

    df = kpdvdb.query()
    df = kpdvdb.query(["MDVP"])
    df = kpdvdb.query(["DIAGNOSES"])
    df = kpdvdb.query(
        ["NORM", "SEX", "Fo", "NATLANG", "VISITDATE"],
        NORM=False,
        SEX="M",
        Fo=[50, 150],
        NATLANG=["English", "Japanese"],
        VISITDATE=[np.datetime64("1993-01-01"), np.datetime64("1994-01-01")],
    )
    df = kpdvdb.query(["DIAGNOSES"], DIAGNOSES=["normal voice", "paralysis"])
    print(df)


def test_files(kpdvdb):
    print(kpdvdb.get_files("ah", ["SEX", "Fo"]))
    print(kpdvdb.get_files("rainbow"))


def test_iter_data(kpdvdb):
    for fs, x in kpdvdb.iter_data("rainbow"):
        pass
    for fs, x, info in kpdvdb.iter_data("ah", auxdata_fields=["SEX", "AGE", "NORM"]):
        pass


def test_query_dx_filter(kpdvdb):

    func = lambda dxs: any(dx.startswith("post") for dx in dxs)

    df = kpdvdb.query(diagnoses_filter=func)
    assert "DIAGNOSES" not in df
    assert len(df)

    df = kpdvdb.query(diagnoses_filter=func, include_diagnoses=True)
    assert "DIAGNOSES" in df
    assert len(df)
    assert all(func(dxs) for dxs in df["DIAGNOSES"])
