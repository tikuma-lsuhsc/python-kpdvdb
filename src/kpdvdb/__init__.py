"""KayPENTAX Disordered Voice Database Reader module"""

__version__ = "0.6.3"

import pandas as pd
from os import path
import numpy as np
from glob import glob
import operator
import nspfile
from collections.abc import Sequence
from typing import Literal, Tuple, List, Union, Iterator, Optional

TaskType = Literal["AH", "RAINBOW"]

ChannelType = Union[str, int, Sequence[Union[str, int]]]

DataField = Literal[
    # fmt:off
    "PAT_ID", "VISITDATE", "FILE AH", "NORM", "FILE RAINBOW", "AGE", "SEX",
    "SMOKE", "NATLANG", "ORIGIN", "Fo", "To", "Fhi", "Flo", "STD", "PFR", "Fftr",
    "Fatr", "Tsam", "Jita", "Jitt", "RAP", "PPQ", "sPPQ", "vFo", "ShdB", "Shim",
    "APQ", "sAPQ", "vAm", "NHR", "VTI", "SPI", "FTRI", "ATRI", "DVB", "DSH",
    "DUV", "NVB", "NSH", "NUV", "SEG", "PER",
    # fmt:on
]


class KPDVDB:
    def __init__(self, dbdir: str, remove_unknowns: bool = False):
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
        * Dataframe ID is set to the common prefix of AH and RAINBOW files

        """

        if self._dir == dbdir:
            return

        # gather all files: name as key and NORM/PATHOL as value
        file2df = lambda task: pd.DataFrame(
            [
                [s[-1][:5], s[-1], s[-3] == "NORM"]
                for s in (
                    p.split(path.sep)
                    for p in operator.iconcat(
                        *(
                            glob(path.join(dbdir, vtype, task, "*.NSP"))
                            for vtype in ("NORM", "PATHOL")
                        )
                    )
                )
            ],
            columns=["ID", "FILE", "NORM"],
        ).set_index("ID")

        files = file2df("AH").join(
            file2df("RAINBOW"), how="outer", lsuffix=" AH", rsuffix=" RAINBOW"
        )
        files["NORM"] = files["NORM AH"].where(
            files["NORM AH"].notna(), files["NORM RAINBOW"]
        )

        df_all = pd.read_table(
            path.join(dbdir, "EXCEL50", "TEXT", "KAYCDALL.TXT"),
            skipinitialspace=True,
            dtype={
                # fmt: off
                **{col: 'Int32' for col in ["AGE", "#", "NVB", "NSH", "NUV", "SEG", "PER"]},
                **{col: "string" for col in ["PAT_ID", "FILE VOWEL 'AH'", "LOCATION"]},
                **{col: "category" for col in ["SEX", "NATLANG", "ORIGIN", "DIAGNOSIS"]},
                **{col: float for col in ["Fo", "To", "Fhi", "Flo", "STD", "PFR", "Fftr", "Fatr", "Tsam", "Jita", 
                                        "Jitt", "RAP", "PQ", "sPPQ", "vFo", "ShdB", "Shim", "APQ", "sAPQ", 
                                        "vAm", "NHR", "VTI", "SPI", "FTRI", "ATRI", "DVB", "DSH", "DUV"]},
                "SMOKE": 'boolean'
                # fmt: on
            },
            parse_dates=["VISITDATE"],
            true_values=["Y"],
            false_values=["N"],
            date_format=["VISITDATE"],
        )

        # drop the columns related to diagnosis
        df = df_all.groupby("FILE VOWEL 'AH'")[
            df_all.columns.drop(["#", "DIAGNOSIS", "LOCATION"])
        ].first()

        # reindex using the first 5 letters of the wav files
        df.index = df.index.str[:5]
        df.index.name = "ID"

        # merge the rainbow files
        df = df.merge(
            files[["NORM", "FILE RAINBOW", "FILE AH"]],
            "left" if remove_unknowns else "outer",
            left_on="FILE VOWEL 'AH'",
            right_on="FILE AH",
        ).drop(columns=["FILE VOWEL 'AH'"])

        if remove_unknowns:
            # drop all entries with unknown PAT_ID (incl. norm)
            df = df[df["PAT_ID"].notna()]

        # add PAT_ID if missing (use unique ID based on the file name)
        df["ID"] = df["FILE AH"].str[:5]
        tf = df["ID"].isna() & df["FILE RAINBOW"].notna()
        df.loc[tf, "ID"] = df.loc[tf, "FILE RAINBOW"].str[:5]
        df = df.set_index("ID").sort_index()

        # split dx
        df_dx = df_all.set_index("FILE VOWEL 'AH'")[["DIAGNOSIS", "LOCATION"]]
        df_dx = df_dx[df_dx["DIAGNOSIS"].notna()]  # drop empty

        # update the df_dx to get the ID instead of the file name
        df_dx = (
            df_dx.merge(df["FILE AH"], "left", left_index=True, right_on="FILE AH")
            .drop(columns="FILE AH")
            .sort_index()
        )

        self._dir = dbdir
        self._df = df
        self._df_dx = df_dx
        self._dx = None

    def get_fields(self) -> List[str]:
        """get list of all database fields

        :return: list of field names
        :rtype: List[str]
        """
        return sorted([*self._df.columns.values, "DIAGNOSES"])

    def get_sexes(self) -> List[str]:
        """get unique entries of SEX field

        :return: fixed list ["F", "M"]
        :rtype: List[str]
        """
        return self._df["SEX"].cat.categories.tolist()

    def get_natlangs(self) -> List[str]:
        """get unique entries of NATLANGS field

        :return: list of subjects' native languages
        :rtype: List[str]
        """
        return self._df["NATLANG"].cat.categories.tolist()

    def get_origins(self) -> List[str]:
        """get unique entries of ORIGIN field

        :return: list of subjects' races
        :rtype: List[str]
        """
        return self._df["ORIGIN"].cat.categories.tolist()

    def get_diagnoses(self, include_locations=False) -> List[str]:
        """get unique entries of DIAGNOSIS field

        :param include_locations: True to also return diagnosis locations, defaults to False
        :type include_locations: bool, optional
        :return: list of diagnoses
        :rtype: List[str]
        """

        if include_locations:
            return self._df_dx[["DIAGNOSIS", "LOCATION"]].drop_duplicates()
        return self._df_dx["DIAGNOSIS"].cat.categories.tolist()

    def _get_dx_series(self):
        if self._dx is None:  # one-time operation
            self._dx = (
                self._df_dx["DIAGNOSIS"].groupby("ID").apply(lambda x: x.tolist())
            )
        return self._dx

    def query(
        self,
        columns: List[DataField] = None,
        include_diagnoses: bool = False,
        diagnoses_filter: bool = None,
        **filters,
    ) -> pd.DataFrame:
        """query database

        :param columns: database columns to return, defaults to None
        :type columns: List[DataField], optional
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
                df = df[
                    s.map(lambda v: isinstance(v, list) and len(fcond & set(v)) > 0)
                ]
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
            df = df[
                [isinstance(v, list) and diagnoses_filter(v) for v in df["DIAGNOSES"]]
            ]

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
        task: TaskType,
        auxdata_fields: List[DataField] = None,
        diagnoses_filter: List[str] = None,
        **filters,
    ) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
        """get NSP filepaths

        :param task: utterance task
        :type task: "rainbow" or "ah"
        :param auxdata_fields: names of auxiliary data fields to return, defaults to None
        :type auxdata_fields: sequence of str, optional
        :param diagnoses_filter: Function with the signature:
                                    diagnoses_filter(diagnoses: List[str]) -> bool
                                 Return true to include the database row to the query
        :type diagnoses_filter: Function
        :param **filters: query conditions (values) for specific per-database columns (keys)
        :type **filters: dict
        :return: list of NSP files and optionally
        :rtype: List[str] or tuple(List[str], pandas.DataFrame)

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
                "ah": ("FILE AH", "AH"),
            }[task]
        except:
            raise ValueError(
                f'Unknown voice task: {task} (must be either "rainbow" or "ah")'
            )

        columns = [col, "NORM"]
        if auxdata_fields is not None:
            columns.extend(auxdata_fields)

        df = self.query(
            columns,
            diagnoses_filter=diagnoses_filter,
            **filters,
        )
        df = df[df.loc[:, columns[0]].notna()]  # drop entries w/out file
        fdf = df.loc[:, columns[:2]]
        files = [
            path.join(self._dir, "NORM" if isnorm else "PATHOL", subdir, f)
            for f, isnorm in zip(fdf[col], fdf["NORM"])
        ]

        return (
            files
            if auxdata_fields is None
            else (files, df.reset_index()[auxdata_fields])
        )

    def iter_data(
        self,
        task: TaskType,
        channels: ChannelType = None,
        auxdata_fields: List[DataField] = None,
        normalize: bool = True,
        diagnoses_filter: List[str] = None,
        **filters,
    ) -> Iterator[Tuple[str, np.array, Optional[pd.Series]]]:
        """iterate over data samples

        :param task: utterance task
        :type task: "rainbow" or "ah"
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
            task,
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

    def read_data(
        self,
        id: str,
        task: TaskType = None,
        channels: ChannelType = None,
        normalize: bool = True,
    ) -> Tuple[int, np.array]:
        """read audio data of one voice task of a recording session

        :param id: recording id
        :type id: str
        :param task: voice task, defaults to None
        :type task: TaskType, optional
        :param channels: audio channel, defaults to None
        :type channels: ChannelType, optional
        :param normalize: True to normalize data between -1 and 1, defaults to True
        :type normalize: bool, optional
        :return: tuple of sampling rate in S/s and data samples numpy array
        :rtype: tuple(int,np.array)
        """
        file = self.get_files(task, ID=id)[0]
        return self._read_file(file, channels, normalize)

    def _read_file(self, file, channels=None, normalize=True):
        fs, x = nspfile.read(file, channels)
        if normalize:
            x = x / 2.0**15
        return fs, x

    def __getitem__(self, key: str) -> Tuple[str, np.array]:
        return self.read_data(key)
