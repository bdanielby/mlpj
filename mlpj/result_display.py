"""
a class with generators that can be used to put results on HTML pages
"""
import os
import sys
import re
import logging
import contextlib
import time

import jinja2
import markupsafe
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import lockfile

from mlpj import pandas_utils as pdu
from mlpj import python_utils as pu


class HTMLDisplay(object):
    """
    * {pa key} in {⠠d savefig, print, printer}
    * If the key contains a colon, the part before the colon is used as the HTML
    file without the extensions instead of "index" and the part after the colon
    is used as the proper key.
    """
    def __init__(self, db_path, project_name, html_dir, image_dir,
        default_figsize=(8, 6), further_html_headers='',
        # Let the Javascript refresh run at most for 1800 sec.
        refresh_how_long=30 * 60, # 30 minutes,
        refresh_not_started_after=12 * 3600, # 12 hours
    ):
        self.db_path = os.path.abspath(db_path)

        with open(
                os.path.join(os.path.dirname(__file__, "result_template.html"))
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
        
        self.html_index_filename = 'index.html'
        self.default_figsize = default_figsize
        self.further_html_headers = further_html_headers
        self.refresh_how_long = refresh_how_long
        self.refresh_not_started_after = refresh_not_started_after
    
    def _init_db_if_necessary(self):
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

    def add_db_entry(self, key, contents, suppl=False):
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
            
    def get_keys(self):
        """Get all distinct keys from the database, reverse-ordered by
        timestamp.
        
        Returns:
            list of str: list of keys
        """
        with pu.sqlite3_conn(self.db_path) as (db, cursor):
            return pu.first_of_each_item(
                cursor.execute('select distinct key from findings '
                               'order by timestamp desc'))

    def del_keys(self, keys):
        """Delete the given keys from the database.

        Args:
            keys (list of str): keys to delete
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

    def del_keys_like(self, regex):
        """Delete keys matching the passed regex from the database.

        Args:
            regex (str): regular expression for the keys
        """
        if pu.isstring(regex):
            regex = re.compile(regex)
        selected_keys = []
        for key in self.get_keys():
            if regex.search(key):
                selected_keys.append(key)
        self.del_keys(selected_keys)

    def _get_image_filepath(self, filename_stem):
        """Add the image directory and ".png" to the filename stem.

        Args:
            filename_stem (str): filename without directory or extension
        Returns:
            corresponding image filepath
        """
        filename = filename_stem + '.png'
        return os.path.join(self.image_dir, filename)
        
    def _regenerate_html(self, cursor, descending=True):
        """Regenerate the result HTML page.

        Args:
            cursor: Sqlite3 database cursor
            descending (bool): whether to sort in descending timestamp order
        """
        all_entries = {}
        # The subquery is necessary for the reverse ordering by timestamp within
        # the groups defined by the keys.
        if descending:
            desc_part = 'desc'
        else:
            desc_part = ''
            
        for key, contents, _ in cursor.execute(f"""
            select key, group_concat(contents, "\n"), max(timestamp) as maxts from (
            select key, contents, timestamp from findings
            order by timestamp desc, ind)
            group by key order by maxts {desc_part}"""
        ):
            contents = re.sub(r'<img src="([^"]*)"',
                              r'<img data-src="\1" class="lazy"', contents)
            
            key_parts = key.split(':', maxsplit=1)
            if len(key_parts) == 1:
                proper_key = key
                html_filename = self.html_index_filename
            else:
                html_filename, proper_key = key_parts
                html_filename += '.html'

            html_filepath = os.path.join(self.html_dir, html_filename)
            existing_entries = all_entries.get(html_filepath, [])
            
            existing_entries.append((proper_key, contents))
            all_entries[html_filepath] = existing_entries
            
        for html_filepath, entries in all_entries.items():
            entries = [
                (key, markupsafe.Markup(value)) for key, value in entries]
            with pu.open_overwriting_safely(html_filepath, 'w') as fout:
                fout.write(
                    self.html_template.render(
                        display=self, entries=entries,
                        further_html_headers=self.further_html_headers,
                        html_filepath=html_filepath
                ))

    RE_VALID_FIGURE_KEY = re.compile(r'^[^\s/()\[\]]+$')

    def _get_figure_path(self, key):
        if self.RE_VALID_FIGURE_KEY.match(key) is None:
            raise ValueError(
                'Please avoid whitespace, parentheses or slashes in the '
                f'key for savefig: "{key}"')
        plot_filepath = self._get_image_filepath(key)
        return key, plot_filepath

    @contextlib.contextmanager
    def savefig(self, key, description='', close_all=True,
                with_printer=True, tool='matplotlib',
                with_bystyle=True, figsize=None, tight_layout=True,
                logging_level=None, refresh_millisec=None):
        """
        The path entry 'image' is expected to exist. The database indexing the
        plots is automatically updated.

        The extension of the image key can be omitted. ".png" will then be
        added.

        The given key is always prepended to the description.

        If there's an exception in the user block, the old entry in the database
        and the HTML log will remain untouched.
            
        :param tool: one of ['matplotlib' | 'r' | 'system']
        :type tool: str

        A ``return`` statement in the block prevents the inclusion of the plot,
        because it is executed only if the loop is exited normally.

        * {pa with_printer}: whether to include the functionality of
        {⠠d printer}
          * A Python logger including the logged contents in the HTML log will
          also be temporarily active.
        """
        key, plot_filepath = self._get_figure_path(key)

        # Make the image creation atomic by using a different filename and
        # moving the image in the end. See the {fu os.rename} call in the end.
        orig_plot_filepath = plot_filepath
        plot_filepath = re.sub(r'(\.[^.]+)$', r'.part\1', plot_filepath)
        
        if figsize is None:
            figsize = self.default_figsize
        elif tool not in ('r', 'matplotlib'):
            raise NotImplementedError('figsize for system')

        if tool == 'matplotlib':
            plt.figure(1, figsize=figsize)

        if with_printer:
            out = pu.StringIOToleratingStr()
            branched = pu.BranchedOutputStreams((out, sys.stdout))
            text = None
            with pu.redirect_stdouterr(branched, branched):
                with pdu.wide_display():
                    #    self._logging_wrapper(branched, logging_level,
                    #                          branched=False):
                    try:
                        #if tool == 'r':
                        #    self._plot_in_r(plot_filepath, figsize)
                        if tool == 'matplotlib':
                            if with_bystyle:
                                from mlpj import plots
                                with plots.bystyle_context():
                                    yield plot_filepath
                            else:
                                yield plot_filepath
                            if tight_layout:
                                self._tight_layout()
                        else:
                            yield plot_filepath
                    finally:
                        text = out.getvalue().strip()
                        if text:
                            if description:
                                description += '\n'
                            description += text
        else:
            #if tool == 'r':
            #    self._plot_in_r(plot_filepath, pixelsize_args)
            if tool == 'matplotlib':
                if with_bystyle:
                    from mlpj import plots
                    with plots.bystyle_context():
                        yield plot_filepath
                else:
                    yield plot_filepath
                if tight_layout:
                    self._tight_layout()
            else:
                yield plot_filepath
        print(f"### new plot arrived: {key}")
        if tool == 'matplotlib':
            plt.savefig(plot_filepath)
        if description:
            print(description)
            description += '\n'

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
        contents = (f'<img src="{path}" id={key}><pre>{description}</pre>'
                    f'{refresh_code}')
            
        self.add_db_entry(key, contents)
        if close_all:
            plt.close('all')

    def print(self, key, *content, **kwargs):
        suppl = kwargs.pop('suppl', False)
        print('####', key, end=' ')
        print(*content, **kwargs)
        out = pu.StringIOToleratingStr()
        kwargs['file'] = out
        print(*content, **kwargs)
        output = out.getvalue().rstrip()
        contents = f"<pre>{output}</pre>"
        self.add_db_entry(key, contents, suppl=suppl)

    def _tight_layout(self):
        from mlpj import plot_utils
        try:
            if not plot_utils.AVOID_TIGHT_LAYOUT:
                plt.tight_layout(pad=0.3)
        except ValueError:
            pass

    def _add_print(self, key, out, suppl, silence_stdout, preformatted):
        text = out.getvalue().rstrip()
        if not text.strip():
            return
        if not silence_stdout:
            print(text)
        if preformatted:
            text = f"<pre>{text}</pre>"
        self.add_db_entry(key, text, suppl=suppl)

    @contextlib.contextmanager
    def printer(self, key, suppl=False, silence_stdout=False, preformatted=True,
              logging_level=None):
        """
        * If there's an exception in the user block, what has been printed
        already will also be mirrored in the database and the HTML log.
        * A Python logger including the logged contents in the HTML log will
        also be temporarily active.
        """
        out = pu.StringIOToleratingStr()
        branched = out
        #pu.BranchedOutputStreams((out, sys.stdout))
        text = None
        try:
            with pu.redirect_stdouterr(branched, branched):
                with pdu.wide_display():
                    yield key
        finally:
            self._add_print(key, out, suppl, silence_stdout, preformatted)

    def link_text(self, description, filepath):
        filepath = pu.make_path_relative_to(filepath, self.html_dir)
        return f"""
</pre>
<a target="_blank" href="{filepath}">{description}</a>
<pre>"""