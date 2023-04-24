"""
Unit tests for `mlpj.result_display`.
"""
import os
import collections
import io

import numpy as np
import pytest

from mlpj import python_utils as pu
from mlpj import project_utils
from mlpj import plot_utils as pltu
from mlpj.project_utils import temp_project


def get_keys_and_contents(pj):
    return [(key, contents) for (key, ind, timestamp, contents) in pj.get_findings()]


def test_printer():
    with temp_project() as pj:
        with pj.printer("key1"):
            print("foo")
            
        with pj.printer("key2"):
            print("bar")
        
        assert pj.get_keys() == ['key2', 'key1']
        assert get_keys_and_contents(pj) == [
            ('key2', '<pre>bar</pre>'),
            ('key1', '<pre>foo</pre>'),
        ]

        with pj.printer("key3"):
            print("bar3")
            
        pj.del_keys('key1')
        assert pj.get_keys() == ['key3', 'key2']

        pj.del_keys_like('3')
        pj.print("key10", "value10")
        assert pj.get_keys() == ['key10', 'key2']

        with pj.printer("key2", suppl=True):
            print("continuation")
            
        assert get_keys_and_contents(pj) == [
            ('key2', '<pre>continuation</pre>'),
            ('key10', 'value10'),
            ('key2', '<pre>bar</pre>'),
        ]


def test_savefig():
    with temp_project() as pj:
        with pj.savefig("profile_plot"):
            n = 500
            x = np.random.random(size=n)
            y = x * x + 0.1 * np.random.random(size=n)

            pltu.profile_plot(x, y, n_bins=30, histogram=None)
            print("profile plot of a noisy x^2")

        assert get_keys_and_contents(pj) == [
            ('profile_plot',
             '<img src="../image/profile_plot.png">'
             '<pre>profile plot of a noisy x^2\n</pre>')
        ]

        assert os.path.exists(os.path.join(pj.image_path, "profile_plot.png"))


def test_link_text():
    with temp_project() as pj:
        assert pj.link_text(
            os.path.join(pj.image_path, "myfile.txt"), 'myfile') == (
                '<a target="_blank" href="../image/myfile.txt">myfile</a>')


def test_print_link_and_return_filepath():
    with temp_project() as pj:
        out = io.StringIO()
        with pu.redirect_stdouterr(out, out):
            filepath = pj.print_link_and_return_filepath(
                'myfile.txt', 'See: ')

        assert out.getvalue() == (
            'See: <a target="_blank" href="../image/myfile.txt">'
            'myfile.txt</a>\n')
        assert filepath == os.path.join(pj.image_path, 'myfile.txt')


def test_get_analysis_pdf_filepath():
    with temp_project() as pj:
        for iteration in [0, 1, -1]:
            if iteration == -1:
                suffix = ''
            else:
                suffix = f'_{iteration}'
            assert (
                pj.get_analysis_pdf_filepath('mymodel', iteration=iteration) ==
                os.path.join(pj.image_path, f'mymodel{suffix}.pdf'))
