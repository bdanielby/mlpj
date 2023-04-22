"""
Unit tests for `mlpj.python_utils`.
"""
import os
import sys
import collections
import io
import tempfile
import datetime

from mlpj import python_utils as pu


def test_all_except()
    lst = ['x', 'a', 'ba', 'a', 'c']
    assert pu.all_except(lst, ['a', 'c']) == ['x', 'ba']


def test_isstring():
    assert pu.isstring("foo")
    assert pu.isstring(b"foo")
    assert not pu.isstring(5)
    assert not pu.isstring(None)


def test_if_true():
    assert pu.if_true(True, 5) == 5
    assert pu.if_true(False, 5) == ''
    assert pu.if_true(False, 5, default=6) == 6


def test_wi_perc():
    assert pu.wi_perc(6, 10) == (6, 60)
    assert pu.wi_perc(6, 11) == (6, 6 / 11 * 100)


def test_perc_str():
    assert pu.perc_str(6, 10) == "6 (60.00 %)"


def test_first_of_each_item():
    assert pu.first_of_each_item([('a', 3, 9), ('b', 4), ('c',)]) == [
        'a', 'b', 'c']
    
    assert (
        pu.first_of_each_item(collections.OrderedDict([('A', 3), ('B', 4)])
                              .items())
        == ['A', 'B'])


def test_mkdir_unless_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        dirpath = os.path.join(tmpdir, "foo", "bar")
        assert not os.path.isdir(dirpath)
        pu.makedir_unless_exists(dirpath)
        assert os.path.isdir(dirpath)
        pu.makedir_unless_exists(dirpath)


def test_make_path_relative_to():
    assert pu.make_path_relative_to('/ab/cd/d/e', '/ab/cd') == 'd/e'
    assert pu.make_path_relative_to('/ab/cd/d/e', '/ab/cd/d2') == '../d/e'
    assert pu.make_path_relative_to('/abd/cd/d/e', '/ab/cd/d2') == (
        '../../../abd/cd/d/e')

    assert pu.make_path_relative_to('/ab//cd//d/e/', '/ab//cd/') == 'd/e'
    assert pu.make_path_relative_to('/ab//cd//d/e/', '/ab//cd//d2') == '../d/e'
    

def test_redirect_stdouterr():
    out = io.StringIO()
    err = io.StringIO()
    with pu.redirect_stdouterr(out, err):
        print("foo")
        print("bar", file=sys.stderr)
    assert out.getvalue() == "foo\n"
    assert err.getvalue() == "bar\n"


def test_BranchedOutputStreams():
    out1 = io.StringIO()
    out2 = io.StringIO()

    stream = pu.BranchedOutputStreams([out1, out2])
    stream.write("foo ")
    stream.flush()
    stream.write("bar")

    for out in (out1, out2):
        assert out.getvalue() == "foo bar"

    stream.close()


def test_open_overwriting_safely():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'foo.txt')
        
        original_output = "foo\nbar\n"
        with open(filepath, 'w') as fout:
            fout.write(original_output)

        def contents(filepath):
            with open(filepath) as fin:
                return fin.read()
            
        new_output = "new output\n"
        try:
            with pu.open_overwriting_safely(filepath, 'w') as fout:
                fout.write(new_output)
                raise ValueError()
        except ValueError:
            pass

        assert contents(filepath) == original_output
        
        with pu.open_overwriting_safely(filepath, 'w') as fout:
            fout.write(new_output)
        assert contents(filepath) == new_output


def test_ansiicol():
    assert pu.ansiicol(31) == "\x1b[31m"


def test_sqlite3_conn():
    with tempfile.TemporaryDirectory() as tmpdir:
        dbpath = os.path.join(tmpdir, "database.db")
        with pu.sqlite3_conn(dbpath) as (db, cursor):
            pass


def test_n_days_ago():
    today = datetime.date.today()
    assert (today - pu.n_days_ago(5)) == datetime.timedelta(days=5)


def test_today_isoformat():
    today = datetime.date.today()
    assert pu.today_isoformat() == today.strftime("%Y-%m-%d")
