"""
Collect textual and numerical results of plots on HTML pages.
"""
import os
import sys
import re
import logging
import time
import io
import sqlite3
import datetime
from typing import Tuple, Optional, List, Union, Any, Callable

import contextlib
import jinja2
import markupsafe
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import lockfile

from . import pandas_utils as pdu
from . import python_utils as pu
from . import plot_utils as pltu


def page_name_and_key_within_page(key: str) -> Tuple[str, str]:
    """Split the result key into the page name and the key within the page

    Args:
        key: result key
    Returns:
       * page name (without '.html' suffix)
       * key within the page
    """
    key_parts = key.split(':', maxsplit=1)
    if len(key_parts) == 1:
        return 'index', key
    else:
        page_name, proper_key = key_parts
        return page_name, proper_key


class HTMLDisplay(object):
    """Collect numerical and textual results and plots on HTML pages.

    Each result is saved under a key in an Sqlite database. Whenever a new
    entry arrives, it is added to the database and the result page is
    regenerated. If a new result arrives for an existing key, the entry is
    updated. The HTML pages contain anchors for each result to make it easy to
    link to them individually.

    To save textual results on page `index.html` under key `mykey` with an
    object `hdisp` of this class use the context manager `printer`:

    `with hdisp.printer('mykey'):
        print('<some result>')`

    Instead of an object of this class, a project manager from module
    `project_utils` can be used because it delegates to this class internally.

    To save a plot on under key `myplot`, use the context manager `savefig`:

    `with hdisp.savefig('myplot'):
        plt.plot(...)`

    The plot will be saved under the name `myplot.png` in the image directory
    and linked on `index.html`.

    If a key contains a colon, the part before the colon is taken as the HTML
    filename to place the result in. (The default is `index(.html)`.)  The part
    after the colon is used as the proper key. The following example will be
    saved on `models.html` under key `prediction_metrics`:

    `with hdisp.printer('models:prediction_metrics'):
        print('<some result>')`

    Args:
        db_path (str): filepath for the backend Sqlite database
        project_name (str): project name
        html_dir (str): dirpath for the generated HTML pages
        image_dir (str): dirpath for the generated plots
        default_figsize (pair of float): default figsize to be used for plots
            (in `matplotlib` and other libraries)
        further_html_headers (str, optional): further headers to add to the
            generated HTML pages
        refresh_how_long (float): upper limit for the number of seconds to run
            the refresh loop (only used if plots request refreshing)
        refresh_not_started_after (float): upper limit for the number of seconds
            after creating the plot to ignore requests for refreshing

    For choosing the page name of the result, you have three options in this
    order of precedence:
    1. Use a key of the format '<page_name>:<key within the page>'.
    2. Use the param `page_name` in the output methods.
    3. Add a string attribute `page_name` for the action.
    """
    def __init__(
        self, db_path: str, project_name: str, html_dir: str, image_dir: str,
        default_figsize: Tuple[float, float] = (8, 6),
        further_html_headers: str = '',
        refresh_how_long: float = pu.SECONDS_IN_HOUR // 2,
        refresh_not_started_after: float = pu.SECONDS_IN_DAY // 2,
    ):
        self.db_path = os.path.abspath(db_path)

        with open(
            os.path.join(os.path.dirname(__file__), "result_template.html")
        ) as fin:
            self.html_template = jinja2.Template(fin.read())

        db_path_dir = os.path.dirname(self.db_path)
        if not os.path.isdir(db_path_dir):
            os.makedirs(db_path_dir)
        self._init_db_if_necessary()

        self.project_name = project_name
        self.db_path = db_path

        self.html_dir = html_dir
        pu.makedir_unless_exists(self.html_dir)

        self.image_dir = image_dir
        pu.makedir_unless_exists(self.image_dir)

        self.default_figsize = default_figsize
        self.further_html_headers = further_html_headers
        self.refresh_how_long = refresh_how_long
        self.refresh_not_started_after = refresh_not_started_after

    def _construct_key(
        self, key: Optional[str], dft_key: Optional[str] = None,
        kpfx: Optional[str] = None, ksfx: Optional[str] = None,
        page_name: Optional[str] = None
    ) -> str:
        """Construct the key for an entry in the result display from the
        argumetns key, kpfx and ksfx
        """
        if key is None:
            assert dft_key is not None
            key = dft_key
            if kpfx is not None:
                key = kpfx + key
            if ksfx is not None:
                key = key + ksfx
        if page_name is not None and page_name != '' and ':' not in key:
            key = f'{page_name}:{key}'
        return key

    @contextlib.contextmanager
    def printer(
        self, key: Optional[str] = None, suppl: bool = False,
        silence_stdout: bool = False, rendering: str = 'preformatted',
        dft_key: Optional[str] = None, kpfx: Optional[str] = None,
        ksfx: Optional[str] = None, page_name: Optional[str] = None
    ) -> None:
        """Context manager to add the output printed in the context manager's
        block to the database under the given key and regenerate the
        corresponding HTML page.

        If there's an exception in the user block, the outputs printed so far
        are preserved.

        Args:
            key (str): result key
            suppl (bool): If `True`, the database entry isn't replaced but
                supplemented.
            silence_stdout (bool): If `True`, the content won't be printed but
                only registered in the database.
            rendering: How to render the output.
                'preformatted': Wrap the output in `<pre>...</pre>`.
                'html': The output is expected to be HTML-formatted. Keep it
                    as is.
                'markdown': the output is expected to be Markdown-formatted.
                    Convert it to HTML.
            dft_key: If key is None, this default key is taken, supplemented by
                kpfx and ksfx if available
            kpfx: prefix for the default key
            ksfx: suffix for the default key
            page_name: The name HTML page (without the suffix '.html') to put
                the results in. Alternatively, the page name can be specified
                within the key like this '<page name>:<key within the page>',
                which also takes precedence ofer page_name.
        """
        key = self._construct_key(key, dft_key, kpfx, ksfx, page_name)

        out = io.StringIO()
        text = None
        try:
            with pu.redirect_stdouterr_branched(
                out, condition=not silence_stdout
            ):
                with pdu.wide_display():
                    yield key
        finally:
            content = out.getvalue().rstrip()
            self.print(key, content, suppl=suppl, silence_stdout=True,
                       rendering=rendering)

    def html(self, *args, **kwargs) -> None:
        """Same as `printer` but with rendering='html'"""
        kwargs['rendering'] = 'html'
        return self.printer(*args, **kwargs)

    def markdown(self, *args, **kwargs) -> None:
        """Same as `printer` but with rendering='markdown'"""
        kwargs['rendering'] = 'markdown'
        return self.printer(*args, **kwargs)

    def _format_output(self, content: str, rendering: str) -> str:
        if rendering == 'preformatted':
            content = f"<pre>{content}</pre>"
        elif rendering == 'markdown':
            import markdown
            content = markdown.markdown(content)
        return content

    def print(
        self, key: str, content: str, suppl: bool = False,
        silence_stdout: bool = False, rendering: str = 'preformatted',
        dft_key: Optional[str] = None, kpfx: Optional[str] = None,
        ksfx: Optional[str] = None, page_name: Optional[str] = None
    ) -> None:
        """Print and add the output to the database under the given key and
        regenerate the corresponding HTML page.

        Do nothing if the content is blank.

        Args:
            key (str): result key
            content (str): to be printed
            suppl (bool): If `True`, the database entry isn't replaced but
                supplemented.
            silence_stdout (bool): If `True`, the content won't be printed but
                only registered in the database.
            rendering: How to render the output.
                'preformatted': Wrap the output in `<pre>...</pre>`.
                'html': The output is expected to be HTML-formatted. Keep it
                    as is.
                'markdown': the output is expected to be Markdown-formatted.
                    Convert it to HTML.
            dft_key: If key is None, this default key is taken, supplemented by
                kpfx and ksfx if available
            kpfx: prefix for the default key
            ksfx: suffix for the default key
            page_name: The name HTML page (without the suffix '.html') to put
                the results in. Alternatively, the page name can be specified
                within the key like this '<page name>:<key within the page>',
                which also takes precedence ofer page_name.
        """
        key = self._construct_key(key, dft_key, kpfx, ksfx, page_name)

        if not content.strip():
            return
        if not silence_stdout:
            print('####', key, content)
        content = self._format_output(content, rendering)
        self.add_db_entry(key, content, suppl=suppl)

    @contextlib.contextmanager
    def savefig(
        self, key: str, tool: str = 'matplotlib', with_printer: bool = True,
        with_libstyle: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        refresh_millisec: Optional[float] = None, tight_layout: bool = True,
        close_all: bool = True, rendering: str = 'preformatted',
        dft_key: Optional[str] = None, kpfx: Optional[str] = None,
        ksfx: Optional[str] = None, page_name: Optional[str] = None
    ) -> None:
        """Context manager to convert the plot created in the context manager's
        block into a PNG file

        The PNG file will be saved under the key in the image directory and
        linked under the key in the corresponding HTML file and the database.
        Printed output will also be added if `with_printer=True`.

        If there's an exception in the user block, the old entry in the database
        and the HTML log will remain untouched.

        Args:
            key (str): result key
            tool ('matplotlib' | 'system'):
                For `matplotlib`, `plt.figure(1, figsize=figsize)` is called
                initially and after executing the block converted to the PNG
                file.

                For `system`, the PNG filepath is available as a with-variable
                for saving the plot under this path.

                Further tools such as Bokeh and Plotly will be supported in the
                future.
            with_printer (bool): If `True`, handle printed output like as in
                the context manager `printer`.
            with_libstyle (bool): If `True`, turn on this library's `matplotlib`
                plot style for the block, see `plot_utils.libstyle`.
            figsize (pair of floats): plot size
            refresh_millisec (float, optional): If a number is given, the HTML
                page will start Javascript timer to refresh the image
                periodically.
            tight_layout (bool): If `True`, call `plt.tight_layout()` in the
                end.
            close_all (bool): If `True`, call `plt.close('all')` in the end.

            suppl (bool): If `True`, the database entry isn't replaced but
                supplemented.
            silence_stdout (bool): If `True`, the content won't be printed but
                only registered in the database.
            rendering: How to render the output.
                'preformatted': Wrap the output in `<pre>...</pre>`.
                'html': The output is expected to be HTML-formatted. Keep it
                    as is.
                'markdown': the output is expected to be Markdown-formatted.
                    Convert it to HTML.
            dft_key: If key is None, this default key is taken, supplemented by
                kpfx and ksfx if available
            kpfx: prefix for the default key
            ksfx: suffix for the default key
            page_name: The name HTML page (without the suffix '.html') to put
                the results in. Alternatively, the page name can be specified
                within the key like this '<page name>:<key within the page>',
                which also takes precedence ofer page_name.
        """
        key = self._construct_key(key, dft_key, kpfx, ksfx, page_name)

        key, plot_filepath = self._get_figure_path(key)

        # Make the image creation atomic by using a different filename and
        # moving the image in the end. See the {fu os.rename} call in the end.
        orig_plot_filepath = plot_filepath
        plot_filepath = re.sub(r'(\.[^.]+)$', r'.part\1', plot_filepath)
        description = ""

        if figsize is None:
            figsize = self.default_figsize
        elif tool not in ('r', 'matplotlib'):
            raise NotImplementedError('figsize for system')

        old_backend = matplotlib.get_backend()

        def prepare_matplotlib() -> None:
            if old_backend != 'Agg':
                matplotlib.use('Agg')
            plt.figure(1, figsize=figsize)

        try:
            if with_printer:
                out = io.StringIO()
                with pu.redirect_stdouterr_branched(
                        out, condition=not silence_stdout):
                    with pdu.wide_display():
                        try:
                            if tool == 'matplotlib':
                                prepare_matplotlib()
                                if with_libstyle:
                                    with pltu.libstyle():
                                        yield plot_filepath
                                else:
                                    yield plot_filepath
                                if tight_layout:
                                    self._tight_layout()
                            else:
                                yield plot_filepath
                        finally:
                            description += out.getvalue()
            else:
                #if tool == 'r':
                #    self._plot_in_r(plot_filepath, pixelsize_args)
                if tool == 'matplotlib':
                    prepare_matplotlib()
                    if with_libstyle:
                        with pltu.libstyle():
                            yield plot_filepath
                    else:
                        yield plot_filepath
                    if tight_layout:
                        self._tight_layout()
                else:
                    yield plot_filepath
            print(f"### new plot arrived: {key}")

        finally:
            if tool == 'matplotlib':
                plt.savefig(plot_filepath)
                matplotlib.rcParams['backend'] = old_backend

        # atomic creation of the image (important for refresh)
        os.rename(plot_filepath, orig_plot_filepath)
        plot_filepath = orig_plot_filepath

        path = pu.make_path_relative_to(plot_filepath, self.html_dir)

        refresh_code = ""
        if refresh_millisec is not None:
            # {p how_long, end_time} are measured in seconds, not milliseconds
            end_time= int(time.time()) + self.refresh_not_started_after

            refresh_code = f"""
            <script>
            start_image_refresh_timer(
              {self.refresh_how_long}, {end_time}, '{key}', '{path}',
              {refresh_millisec});
            </script>
            """

        description = self._format_output(description, rendering)
        contents = (f'<img src="{path}"> {description}{refresh_code}')

        self.add_db_entry(key, contents)
        if close_all:
            plt.close('all')

    def add_db_entry(
        self, key: str, contents: str, suppl: bool = False
    ) -> None:
        """Add an entry to the Sqlite3 database file for the given key.

        After adding the entry, regenerate the result HTML page.

        Args:
            key (str): result key
            contents (str): result string (HTML)
            suppl (bool): whether to supplement an existing entry
        """
        with lockfile.LockFile(self.db_path), \
                pu.sqlite3_conn(self.db_path) as (db, cursor):
            ind = 0
            if not suppl:
                cursor.execute("delete from findings where key = ?", (key,))
            else:
                result = list(cursor.execute(
                    "select max(ind) from findings where key = ?", (key,)))
                if len(result) > 0:
                    ind = result[0][0]
                    if ind is None:
                        ind = 0
                    ind += 1
            cursor.execute("insert into findings values (?, ?, ?, ?)",
                           (key, ind, time.time(), contents))
            db.commit()
            self._regenerate_html(cursor)

    def regenerate_html(self) -> None:
        """Regenerate the result pages"""
        with lockfile.LockFile(self.db_path), \
                pu.sqlite3_conn(self.db_path) as (_, cursor):
            self._regenerate_html(cursor)

    def link_text(self, filepath: str, link_text: str= '') -> str:
        """HTML text for a link to a given filepath.

        The filepath is made relative to the HTML directory for the link.

        Args:
            filepath (str): filepath for the link
            link_text (str, optional): link text to display, defaults to given
                filepath
        Returns:
            str: HTML link text
        """
        filepath = pu.make_path_relative_to(filepath, self.html_dir)
        return f'<a target="_blank" href="{filepath}">{link_text}</a>'

    def get_keys(self) -> List[str]:
        """Get all distinct result keys from the database, reverse-ordered by
        timestamp.

        Returns:
            list of str: list of result keys
        """
        with pu.sqlite3_conn(self.db_path) as (db, cursor):
            return pu.first_of_each_item(
                cursor.execute('select distinct key from findings '
                               'order by timestamp desc'))

    def del_keys(self, keys: Union[str, List[str]]) -> None:
        """Delete the given result keys from the database.

        Args:
            keys (list of str): result keys to delete
        """
        if pu.isstring(keys):
            keys = [keys]
        with pu.sqlite3_conn(self.db_path) as (db, cursor):
            for key in keys:
                cursor.execute('delete from findings where key = ?', (key,))
                plot_filepath = self._get_image_filepath(key)
                if os.path.exists(plot_filepath):
                    os.remove(plot_filepath)
            db.commit()

    def del_keys_like(self, regex: str) -> None:
        """Delete result keys matching the passed regex from the database.
        Args:
            regex (str): regular expression for the result keys; it is used with
                re.search
        """
        if pu.isstring(regex):
            regex = re.compile(regex)
        selected_keys = []
        for key in self.get_keys():
            if regex.search(key):
                selected_keys.append(key)
        self.del_keys(selected_keys)

    def rename_keys(self, renamer: Callable[[str], str]) -> None:
        """Rename result keys with the given function

        Nothing is done for keys where the renamer function returns the key
        unchanged.

        Args:
            renamer: function to map the keys
        """
        with pu.sqlite3_conn(self.db_path) as (db, cursor):
            for key, in list(cursor.execute('select key from findings')):
                new_key = renamer(key)
                if new_key != key:
                    print(f"renaming key {key} to {new_key}")
                    cursor.execute(
                        'update findings set key = ? where key = ?',
                        (new_key, key))
            db.commit()

    def move_keys_to_page(self, regex: str, page_name: str) -> None:
        """Move keys matching the regex to another HTML page

        Args:
            regex: regular expression to select the keys to move
            page_name: new page name for the selected keys
        """
        if pu.isstring(regex):
            regex = re.compile(regex)

        def renamer(key: str) -> str:
            """Rename old keys to new keys"""
            if regex.search(key) is None:
                return key
            else:
                _, proper_key = page_name_and_key_within_page(key)
                if page_name == '' or page_name == 'index':
                    return proper_key
                else:
                    return f'{page_name}:{proper_key}'

        self.rename_keys(renamer)

    def move_keys_from_page_to_page(
        self, old_page_name: str, new_page_name: str
    ) -> None:
        """Move keys from one page to another

        Args:
            old_page_name: old page whose keys should be moved; can be an empty
                string for the 'index' page
            new_page_name: new page name for those keys
        """
        if old_page_name == '':
            old_page_name = 'index'

        def renamer(key: str) -> str:
            """Rename old keys to new keys"""
            page_name, proper_key = page_name_and_key_within_page(key)
            if page_name == old_page_name:
                if new_page_name == '' or new_page_name == 'index':
                    return proper_key
                else:
                    return f'{new_page_name}:{proper_key}'

        self.rename_keys(renamer)

    def get_findings(self) -> List[Any]:
        """Get the contents of the findings table in the database,
        reverse-ordered by timestamp.

        Returns:
            list of tuples: rows of the database table
        """
        with pu.sqlite3_conn(self.db_path) as (db, cursor):
            return list(cursor.execute(
                'select * from findings order by timestamp desc'))

    def _init_db_if_necessary(self) -> None:
        """Create the Sqlite3 database file and the table "findings" in it
        unless they already exist.
        """
        with pu.sqlite3_conn(self.db_path) as (db, cursor):
            cursor.execute("""
                create table if not exists findings (
                    key text not null,
                    ind integer not null,
                    timestamp integer not null,
                    contents text not null,
                    primary key (key, ind)
                )""")

    def page_name_to_html_filepath(self, page_name: str) -> str:
        """Translate a page name to an HTML filepath

        Args:
            page_name: page name of a result key
        Returns:
            filepath of the corresponding HTML file
        """
        if page_name == '':
            page_name = 'index'
        return os.path.join(self.html_dir, f'{page_name}.html')

    def _regenerate_html(
        self, cursor: sqlite3.Cursor, descending: bool = True
    ) -> None:
        """Regenerate the result HTML page.

        Args:
            cursor: Sqlite3 database cursor
            descending (bool): whether to sort in descending timestamp order
        """
        all_entries = {self.page_name_to_html_filepath('index'): []}

        # The subquery is necessary for the reverse ordering by timestamp within
        # the groups defined by the keys.
        if descending:
            desc_part = 'desc'
        else:
            desc_part = ''

        for key, contents, timestamp in cursor.execute(f"""
            select key, group_concat(contents, "\n"),
            max(timestamp) as maxts from (
            select key, contents, timestamp from findings
            order by timestamp desc, ind)
            group by key order by maxts {desc_part}"""
        ):
            contents = re.sub(r'<img src="([^"]*)"',
                              r'<img data-src="\1" class="lazy"', contents)

            timestr = datetime.datetime.fromtimestamp(timestamp).strftime('%c')

            page_name, proper_key = page_name_and_key_within_page(key)
            html_filepath = self.page_name_to_html_filepath(page_name)

            existing_entries = all_entries.get(html_filepath, [])

            existing_entries.append((proper_key, contents, timestr))
            all_entries[html_filepath] = existing_entries

        for html_filepath, entries in all_entries.items():
            entries = [
                (key, markupsafe.Markup(value), timestr)
                for key, value, timestr in entries]
            with pu.open_overwriting_safely(html_filepath, 'w') as fout:
                fout.write(
                    self.html_template.render(
                        display=self, entries=entries,
                        further_html_headers=self.further_html_headers,
                        html_filepath=html_filepath
                ))

    RE_VALID_FIGURE_KEY = re.compile(r'^[^\s/()\[\]]+$')

    def _get_image_filepath(self, filename_stem: str) -> str:
        """Add the image directory and ".png" to the filename stem.

        Args:
            filename_stem (str): filename without directory or extension
        Returns:
            corresponding image filepath
        """
        filename = filename_stem + '.png'
        return os.path.join(self.image_dir, filename)

    def _get_figure_path(self, key: str) -> str:
        """Check whether the key is valid and determine the filepath for a plot
        file.

        Args:
            key (str): result key
        Returns:
            key, plot filepath
        """
        if self.RE_VALID_FIGURE_KEY.match(key) is None:
            raise ValueError(
                'Please avoid whitespace, parentheses or slashes in the '
                f'key for savefig: "{key}"')
        plot_filepath = self._get_image_filepath(key)
        return key, plot_filepath

    def _tight_layout(self) -> None:
        """Call `plt.tight_layout` and catch `ValueError`s."""
        try:
            plt.tight_layout(pad=0.3)
        except ValueError:
            pass
