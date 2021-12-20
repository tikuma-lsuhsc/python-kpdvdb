"""KayPENTAX Disordered Voice Database Reader module"""

__version__ = "0.1.1"

import pandas as pd
from os import path
import numpy as np
from glob import glob as _glob
import re, operator
import nspfile

# global variables
_dir = None  # database dir
_df = None  # main database table
_df_dx = None  # patient diagnosis table
_dx = None  # patient diagnoses series


def load_db(dbdir):
    """load disordered voice database

    :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
    :type dbdir: str

    * This function must be called at the beginning of each Python session
    * Database is loaded from the text file found at: <dbdir>/EXCEL50/TEXT/KAYCDALL.TXT
    * Only entries with NSP files are included
    * PAT_ID of the entries without PAT_ID field uses the "FILE VOWEL 'AH'" field value
      as the PAT_ID value

    """

    global _dir, _df, _df_dx, _dx

    if _dir == dbdir:
        return

    ra_files = operator.iconcat(
        *(
            _glob(path.join(dbdir, vtype, "RAINBOW", "*.NSP"))
            for vtype in ("NORM", "PATHOL")
        )
    )

    df = pd.read_table(
        path.join(dbdir, "EXCEL50", "TEXT", "KAYCDALL.TXT"),
        skipinitialspace=True,
        dtype={
            # fmt: off
            **{col: 'Int32' for col in ["AGE", "#", "NVB", "NSH", "NUV", "SEG", "PER"]},
            **{col: "string" for col in ["PAT_ID", "FILE VOWEL 'AH'"]},
            **{col: "category" for col in ["SEX", "DIAGNOSIS", "LOCATION", "NATLANG", "ORIGIN"]},
            **{col: float for col in ["Fo", "To", "Fhi", "Flo", "STD", "PFR", "Fftr", "Fatr", "Tsam", "Jita", 
                                    "Jitt", "RAP", "PQ", "sPPQ", "vFo", "ShdB", "Shim", "APQ", "sAPQ", 
                                    "vAm", "NHR", "VTI", "SPI", "FTRI", "ATRI", "DVB", "DSH", "DUV"]},
            "SMOKE": 'boolean'
            # fmt: on
        },
        parse_dates=["VISITDATE"],
        true_values=["Y"],
        false_values=["N"],
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
    df.reset_index(inplace=True, drop=True)

    # assign for rainbow
    ra_names = set((path.basename(f) for f in ra_files))

    # remove entries without PAT_ID or FILE VOWEL 'AH'
    n = df.shape[0]
    rnames = [None] * n
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

    df.insert(3, "FILE RAINBOW", pd.Series(rnames, dtype="string"))
    df.insert(3, "NORM", pd.Series(isnorms, dtype="boolean"))
    df.drop(index=np.where(tf)[0], inplace=True)
    df.reset_index(inplace=True, drop=True)

    _dir = dbdir
    _df = df
    _df_dx = df_dx
    _dx = None


def get_fields():
    """get list of all database fields

    :return: list of field names
    :rtype: list(str)
    """
    return sorted([*_df.columns.values, "DIAGNOSES"])


def get_sexes():
    """get unique entries of SEX field

    :return: fixed list ["F", "M"]
    :rtype: list(str)
    """
    return _df["SEX"].cat.categories.tolist()


def get_locations():
    """get unique entries of LOCATION field

    :return: list of sites of disorders
    :rtype: list(str)
    """
    return _df["LOCATION"].cat.categories.tolist()


def get_natlangs():
    """get unique entries of NATLANGS field

    :return: list of subjects' native languages
    :rtype: list(str)
    """
    return _df["NATLANG"].cat.categories.tolist()


def get_origins():
    """get unique entries of ORIGIN field

    :return: list of subjects' races
    :rtype: list(str)
    """
    return _df["ORIGIN"].cat.categories.tolist()


def get_diagnoses():
    """get unique entries of DIAGNOSIS field

    :return: list of diagnoses
    :rtype: list(str)
    """

    return _df_dx["DIAGNOSIS"].cat.categories.tolist()


def _get_dx_series():
    global _dx
    if _dx is None:  # one-time operation
        _dx = pd.Series(
            [
                _df_dx[(_df_dx["PAT_ID"] == id) & (_df_dx["VISITDATE"] == date)][
                    "DIAGNOSIS"
                ]
                .dropna()
                .tolist()
                for id, date in zip(_df["PAT_ID"], _df["VISITDATE"])
            ]
        )
    return _dx


def query(subset=None, include_diagnoses=False, **filters):
    """query database

    :param subset: database columns to return, defaults to None
    :type subset: sequence of str, optional
    :param include_diagnoses: True to include DIAGNOSES column. Ignored if subset or filters
                              specifies DIAGNOSES, defaults to False
    :type include_diagnoses: bool, optional
    :param **filters: query conditions (values) for specific per-database columns (keys)
    :type **filters: dict
    :return: query result
    :rtype: pandas.DataFrame

    Valid values of `subset` argument (get_fields() + "MDVP")
    ---------------------------------------------------------

    * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
      (except for "DIAGNOSIS" and "#")
    * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
    * "NORM" - True if normal data, False if pathological data
    * "MDVP" - Short-hand notation to include all the MDVP parameter measurements: from "Fo" to "PER"

    Valid `filters` keyword arguments (get_fields())
    ------------------------------------------------

    * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
      (except for "DIAGNOSIS" and "#")
    * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
    * "NORM" - True if normal data, False if pathological data
    
    Valid `filters` keyword argument values
    ---------------------------------------

    * A scalar value
    * For numeric and date columns, 2-element sequence to define a range: [start, end)
    * For all other columns, a sequence of allowable values

    """

    # work on a copy of the dataframe
    df = _df.copy(deep=True)
    if (
        include_diagnoses
        or (subset is not None and "DIAGNOSES" in subset)
        or "DIAGNOSES" in filters
    ):
        df.insert(3, "DIAGNOSES", _get_dx_series())

    # apply the filters to reduce the rows
    for fcol, fcond in filters.items():
        try:
            s = df[fcol]
        except:
            raise ValueError(f"{fcol} is not a valid column label")
        if fcol == "DIAGNOSES":
            fcond = set([fcond]) if isinstance(fcond, str) else set(fcond)
            df = df[[len(fcond & set(v)) > 0 for v in s]]
        else:
            try:  # try range/multi-choices
                if s.dtype.kind in "iufcM":  # numeric/date
                    # 2-element range condition
                    df = df[(s >= fcond[0]) & (s < fcond[1])]
                else:  # non-numeric
                    df = df[s.isin(fcond)]  # choice condition
            except:
                # look for the exact match
                df = df[s == fcond]

    # return only the selected columns
    if subset is not None:
        try:
            i = subset.index("MDVP")
            cols = df.columns
            j = np.where(cols == "Fo")[0][0]
            subset = [*subset[:i], *cols[j:].values, *subset[i + 1 :]]
        except:
            pass
        try:
            df = df[subset]
        except:
            raise ValueError(
                f'At least one label in the "subset" argument is invalid: {subset}'
            )

    return df


def get_files(type, auxdata_fields=None, **filters):
    """get NSP filepaths

    :param type: utterance type
    :type type: "rainbow" or "ah"
    :param other_fields: names of auxiliary data fields to return, defaults to None
    :type other_fields: sequence of str, optional
    :param **filters: query conditions (values) for specific per-database columns (keys)
    :type **filters: dict
    :return: list of NSP files and optionally
    :rtype: list(str) or tuple(list(str), pandas.DataFrame)

    Valid values of `subset` argument
    ---------------------------------

    * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
      (except for "DIAGNOSIS" and "#")
    * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
    * "NORM" - True if normal data, False if pathological data
    * "MDVP" - Short-hand notation to include all the MDVP parameter measurements: from "Fo" to "PER"

    Valid `filters` keyword arguments
    ---------------------------------

    * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
      (except for "DIAGNOSIS" and "#")
    * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
    * "NORM" - True if normal data, False if pathological data

    Valid `filters` keyword argument values
    ---------------------------------------

    * A scalar value
    * For numeric and date columns, 2-element sequence to define a range: [start, end)
    * For all other columns, a sequence of allowable values
    """

    try:
        col, subdir = {
            "rainbow": ("FILE RAINBOW", "RAINBOW"),
            "ah": ("FILE VOWEL 'AH'", "AH"),
        }[type]
    except:
        raise ValueError(f'Unknown type: {type} (must be either "rainbow" or "ah")')

    subset = [col, "NORM"]
    if auxdata_fields is not None:
        subset.extend(auxdata_fields)

    df = query(subset, **filters)
    df = df[df.iloc[:, 0].notna()]  # drop entries w/out file
    fdf = df.iloc[:, :2]
    files = [
        path.join(_dir, "NORM" if isnorm else "PATHOL", subdir, f)
        for f, isnorm in zip(fdf[col], fdf["NORM"])
    ]

    return (
        files
        if auxdata_fields is None
        else (files, df.iloc[:, 2:].reset_index(drop=True))
    )


def iter_data(type, channels=None, auxdata_fields=None, **filters):
    """iterate over data samples

    :param type: utterance type
    :type type: "rainbow" or "ah"
    :param channels: audio channels to read ('a', 'b', 0-1, or a sequence thereof),
                     defaults to None (all channels)
    :type channels: str, int, sequence, optional
    :param other_fields: names of auxiliary data fields to return, defaults to None
    :type other_fields: sequence of str, optional
    :param **filters: query conditions (values) for specific per-database columns (keys)
    :type **filters: dict
    :yield:
        - sampling rate : audio sampling rate in samples/second
        - data  : audio data, 1-D for 1-channel NSP (only A channel), or 2-D of shape
                  (Nsamples, 2) for 2-channel NSP
        - auxdata : (optional) requested auxdata of the data if auxdata_fields is specified
    :ytype: tuple(int, numpy.ndarray(int16)[, pandas.Series])

    Iterates over all the DataFrame columns, returning a tuple with the column name and the content as a Series.

    Yields

        labelobject

            The column names for the DataFrame being iterated over.
        contentSeries

            The column entries belonging to each label, as a Series.



    Valid values of `subset` argument
    ---------------------------------

    * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
      (except for "DIAGNOSIS" and "#")
    * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
    * "NORM" - True if normal data, False if pathological data
    * "MDVP" - Short-hand notation to include all the MDVP parameter measurements: from "Fo" to "PER"

    Valid `filters` keyword arguments
    ---------------------------------

    * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
      (except for "DIAGNOSIS" and "#")
    * "DIAGNOSES" - A list containing all the original "DIAGNOSIS" associated with the subject
    * "NORM" - True if normal data, False if pathological data

    Valid `filters` keyword argument values
    ---------------------------------------

    * A scalar value
    * For numeric and date columns, 2-element sequence to define a range: [start, end)
    * For all other columns, a sequence of allowable values
    """

    files = get_files(type, auxdata_fields, **filters)

    hasaux = auxdata_fields is not None
    if hasaux:
        files, auxdata = files

    for i, file in enumerate(files):
        out = nspfile.read(file, channels)
        yield (*out, auxdata.loc[i, :]) if hasaux else out
