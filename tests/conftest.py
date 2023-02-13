"""
conftest.py
"""

import pytest
import winreg
from os import path

from contextlib import suppress
import itertools

from kpdvdb import KPDVDB


def find_onedrive():
    reg_base = "Software\Microsoft\OneDrive\Accounts"
    found = False
    with suppress(WindowsError), winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, reg_base
    ) as key:
        for i in itertools.count():
            with winreg.OpenKey(key, winreg.EnumKey(key, i)) as acct:
                drive_dir = winreg.QueryValueEx(acct, "UserFolder")[0]
                found = "LSUHSC" in drive_dir
                if found:
                    break
    assert found
    return drive_dir


def load_db(only_known=False):
    dbdir = path.join(find_onedrive(), "data", "KayPENTAX Disordered Voice Database")
    return KPDVDB(dbdir, only_known)


@pytest.fixture(scope="module")
def kpdvdb():
    return load_db()
