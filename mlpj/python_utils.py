"""
Utilities and convenience functions for using the Python standard library.
"""
import os


def makedir_unless_exists(dirpath):
    """Create a directory path unless it already exists.

    Args:
        dirpath (str): directory path
    """
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)


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
    f"\x1b[{color_num}{bright_part}m"
    

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


SECONDS_IN_HOUR = 3600

SECONDS_IN_DAY = 24 * SECONDS_IN_HOUR
