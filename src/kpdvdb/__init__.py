"""KayPENTAX Disordered Voice Database Reader module"""

__version__ = "0.4.0"

import pandas as pd
from os import path
import numpy as np
from glob import glob as _glob
import re, operator
import nspfile


class KPDVDB:
    def __init__(self, dbdir, remove_unknowns=False):
        """KPDVDB constructor

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str
        """

        # database variables
        self._dir = None  # database dir
        self._df = None  # main database table
        self._df_dx = None  # patient diagnosis table
        self._dx = None  # patient diagnoses series

        # load the database
        self._load_db(dbdir, remove_unknowns)

    def _load_db(self, dbdir, remove_unknowns):
        """load disordered voice database

        :param dbdir: path to the cdrom drive or the directory hosting a copy of the database
        :type dbdir: str

        * This function must be called at the beginning of each Python session
        * Database is loaded from the text file found at: <dbdir>/EXCEL50/TEXT/KAYCDALL.TXT
        * Only entries with NSP files are included
        * PAT_ID of the entries without PAT_ID field uses the "FILE VOWEL 'AH'" field value
        as the PAT_ID value

        """

        if self._dir == dbdir:
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
        if remove_unknowns:
            df = df[df["PAT_ID"].notna()]
        else:
            tf = pd.isna(df["PAT_ID"])
            df.loc[tf, "PAT_ID"] = df.loc[tf, "FILE VOWEL 'AH'"].str.slice(stop=-6)

        # split dx
        df_dx = df[["PAT_ID", "VISITDATE", "DIAGNOSIS", "LOCATION"]]
        df.drop(columns=["#", "DIAGNOSIS", "LOCATION"], inplace=True)
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
        df.index.set_names("ID", inplace=True)

        self._dir = dbdir
        self._df = df
        self._df_dx = df_dx[df_dx["DIAGNOSIS"].notna()]
        self._dx = None

    def get_fields(self):
        """get list of all database fields

        :return: list of field names
        :rtype: list(str)
        """
        return sorted([*self._df.columns.values, "DIAGNOSES"])

    def get_sexes(self):
        """get unique entries of SEX field

        :return: fixed list ["F", "M"]
        :rtype: list(str)
        """
        return self._df["SEX"].cat.categories.tolist()

    def get_natlangs(self):
        """get unique entries of NATLANGS field

        :return: list of subjects' native languages
        :rtype: list(str)
        """
        return self._df["NATLANG"].cat.categories.tolist()

    def get_origins(self):
        """get unique entries of ORIGIN field

        :return: list of subjects' races
        :rtype: list(str)
        """
        return self._df["ORIGIN"].cat.categories.tolist()

    def get_diagnoses(self, include_locations=False):
        """get unique entries of DIAGNOSIS field

        :param include_locations: True to also return diagnosis locations, defaults to False
        :type include_locations: bool, optional
        :return: list of diagnoses
        :rtype: list(str)
        """

        if include_locations:
            return self._df_dx[["DIAGNOSIS", "LOCATION"]].drop_duplicates()
        return self._df_dx["DIAGNOSIS"].cat.categories.tolist()

    def _get_dx_series(self):

        if self._dx is None:  # one-time operation
            self._dx = pd.Series(
                [
                    self._df_dx[
                        (self._df_dx["PAT_ID"] == id)
                        & (self._df_dx["VISITDATE"] == date)
                    ]["DIAGNOSIS"]
                    .dropna()
                    .tolist()
                    for id, date in zip(self._df["PAT_ID"], self._df["VISITDATE"])
                ]
            )
        return self._dx

    def query(
        self,
        columns=None,
        include_diagnoses=False,
        diagnoses_filter=None,
        **filters,
    ):
        """query database

        :param columns: database columns to return, defaults to None
        :type columns: sequence of str, optional
        :param include_diagnoses: True to include DIAGNOSES column. Ignored if columns or filters
                                specifies DIAGNOSES, defaults to False
        :type include_diagnoses: bool, optional
        :param diagnoses_filter: Function with the signature:
                                    diagnoses_filter(diagnoses: List[str]) -> bool
                                 Return true to include the database row to the query
        :type diagnoses_filter: Function
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: query result
        :rtype: pandas.DataFrame

        Valid values of `columns` argument (get_fields() + "MDVP")
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
        df = self._df.copy(deep=True)
        incl_dx = (
            include_diagnoses
            or (columns is not None and "DIAGNOSES" in columns)
            or "DIAGNOSES" in filters
        )

        if incl_dx or diagnoses_filter:
            df.insert(3, "DIAGNOSES", self._get_dx_series())

        # apply the filters to reduce the rows
        for fcol, fcond in filters.items():
            try:
                if fcol == "ID":
                    s = df.index
                else:
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

        # if diagnosis screening function is given, further filter the rows
        if diagnoses_filter:
            df = df[[diagnoses_filter(v) for v in df["DIAGNOSES"]]]

            if not incl_dx:
                # if dx not requested, drop it
                df.drop("DIAGNOSES", axis=1, inplace=True)

        # return only the selected columns
        if columns is not None:
            try:
                i = columns.index("MDVP")
                cols = df.columns
                j = np.where(cols == "Fo")[0][0]
                columns = [*columns[:i], *cols[j:].values, *columns[i + 1 :]]
            except:
                pass
            try:
                df = df[columns]
            except:
                ValueError(
                    f'At least one label in the "columns" argument is invalid: {columns}'
                )

        return df

    def get_files(
        self,
        type,
        auxdata_fields=None,
        diagnoses_filter=None,
        **filters,
    ):
        """get NSP filepaths

        :param type: utterance type
        :type type: "rainbow" or "ah"
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param diagnoses_filter: Function with the signature:
                                    diagnoses_filter(diagnoses: List[str]) -> bool
                                 Return true to include the database row to the query
        :type diagnoses_filter: Function
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: list of NSP files and optionally
        :rtype: list(str) or tuple(list(str), pandas.DataFrame)

        Valid values of `auxdata_fields` argument
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")
        * "ID" - recording ID returned by query()
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

        columns = [col, "NORM"]
        if auxdata_fields is not None:
            columns.extend(auxdata_fields)

        df = self.query(
            columns,
            diagnoses_filter=diagnoses_filter,
            **filters,
        )
        df = df[df.iloc[:, 0].notna()]  # drop entries w/out file
        fdf = df.iloc[:, :2]
        files = [
            path.join(self._dir, "NORM" if isnorm else "PATHOL", subdir, f)
            for f, isnorm in zip(fdf[col], fdf["NORM"])
        ]

        return (
            files
            if auxdata_fields is None
            else (files, df.iloc[:, 2:].reset_index(drop=True))
        )

    def iter_data(
        self,
        type,
        channels=None,
        auxdata_fields=None,
        normalize=True,
        diagnoses_filter=None,
        **filters,
    ):
        """iterate over data samples

        :param type: utterance type
        :type type: "rainbow" or "ah"
        :param channels: audio channels to read ('a', 'b', 0-1, or a sequence thereof),
                        defaults to None (all channels)
        :type channels: str, int, sequence, optional
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param normalize: True to return normalized f64 data, False to return i16 data, defaults to True
        :type normalize: bool, optional
        :param diagnoses_filter: Function with the signature:
                                    diagnoses_filter(diagnoses: List[str]) -> bool
                                 Return true to include the database row to the query
        :type diagnoses_filter: Function
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



        Valid values of `auxdata_fields` argument
        ---------------------------------

        * All columns of the database specified in EXCEL50/TEXT/README.TXT Section 3.1
        (except for "DIAGNOSIS" and "#")
        * "ID" - recording ID returned by query()
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

        files = self.get_files(
            type,
            auxdata_fields,
            diagnoses_filter=diagnoses_filter,
            **filters,
        )

        hasaux = auxdata_fields is not None
        if hasaux:
            files, auxdata = files

        for i, file in enumerate(files):
            out = self._read_file(file, channels, normalize)
            yield (*out, auxdata.loc[i, :]) if hasaux else out

    def read_data(self, id, type=None, channels=None, normalize=True):
        file = self.get_files(type, ID=id)[0]
        return self._read_file(file, channels, normalize)

    def _read_file(self, file, channels=None, normalize=True):
        fs, x = nspfile.read(file, channels)
        if normalize:
            x = x / 2.0**15
        return fs, x
