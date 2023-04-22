"""
Utilities and convenience functions for using the Python standard library.
"""
import os
import sys
import sqlite3
import datetime
import contextlib


def makedir_unless_exists(dirpath):
    """Create a directory path unless it already exists.

    Args:
        dirpath (str): directory path
    """
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)


@contextlib.contextmanager
def redirect_stdouterr(outfp, errfp):
    """Context manager for temporary redirection of `sys.stdout` and
    `sys.stderr`.
    
    Args:
        outfp (output stream): output stream to redirect the standard output
            stream to (default=`os.devnull`)
        errfp (output stream): path or output stream to redirect
            the standard error stream to (default=`os.devnull`)
    
    If you want to keep one of the two unchanged, just pass `sys.stdout` or
    `sys.stderr`, respectively.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = outfp
        sys.stderr = errfp
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        
def isstring(s):
    """Is the argument a string?

    Args:
        s: object
    Returns:
        bool: whether it is of type `str` or `bytes`
    """
    return isinstance(s, (str, bytes))



def ansiicol(color_num, is_bright=False):
    """ANSI escape code for the given color number
    https://en.wikipedia.org/wiki/ANSI_escape_code

    Args:
        color_num (int): color number
        is_bold (bool): whether to use bold face
    Returns:
        ANSII escape code
    """
    if is_bright:
        bright_part = ';1'
    else:
        bright_part = ''
    return f"\x1b[{color_num}{bright_part}m"
    

TERMSEQ = {
    'red[': ansiicol(31),
    'br_red[': ansiicol(31, True),
    'green[': ansiicol(32),
    'br_green[': ansiicol(32, True),
    'yellow[': ansiicol(33),
    'br_yellow[': ansiicol(33, True),
    'cyan[': ansiicol(36),
    'br_cyan[': ansiicol(36, True),
    'blue[': ansiicol(34),
    'br_blue[': ansiicol(34, True),
    #'bold[': "\x1b[30;1m",
    ']': "\x1b[0m",
}


@contextlib.contextmanager
def sqlite3_conn(filepath):
    """Context manager for a connection to a Sqlite3 database file

    Args:
        filepath (str): filepath to the database file
    Yielding:
        db, cursor; the database connection and the cursor
    """
    db = None
    cursor = None
    try:
        db = sqlite3.connect(filepath)
        cursor = db.cursor()
        yield db, cursor
    finally:
        if cursor is not None:
            cursor.close()
        if db is not None:
            db.close()


SECONDS_IN_HOUR = 3600

SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR

    
def n_days_ago(n_days):
    """The datetime object for the day n_days days ago

    Args:
        n_days (int): number of days to go into the past
    Returns:
        `datetime.datetime`: datetime object for that day in the past
    """
    return datetime.date.today() - datetime.timedelta(n_days)
