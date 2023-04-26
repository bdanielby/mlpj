"""
Project management utilities
"""
from __future__ import print_function, division

import os
import sys
import re
import random
import contextlib
import tempfile
from typing import Optional, Any, List, Union
from collections.abc import Generator

import numpy as np

from . import python_utils as pu
from . import result_display, actions_looper


class Manager(object):
    """Creates and delegates to specialist objects for actions looping, result
    display and other things.

    Args:
        project_path (str): directory for the project data; the basename will be
            taken as the project name
        seed (int, optional): random seed, will be registered in the libraries
            `numpy` and `random`
        doc (str, optional): project documentation to be displayed in the command
            line help

    Directory and file structure for the project:
    * `<project_path>/steps`: the persisted interim results of processing
      steps handled by `actions_looper` commands
    * `<project_path>/image`: the plots created with `savefig`
    * `<project_path>/html`: the HTML files generated by `result_display`
      commands like (`savefig`, `printer`), linking to the images in the image
      directory above
    * `<project_path>/html/index.html`: the default HTML page generated by
      `result_display` commands
    * `<project_path>/contents.db`: Sqlite database used by `result_display`
      commands
    """
    def __init__(self, project_path: str, seed: Optional[int] = None, doc: Optional[str] = None):
        self.project_dir = os.path.abspath(os.path.expanduser(project_path))
        pu.makedir_unless_exists(self.project_dir)
        self.name = os.path.basename(self.project_dir)
        
        self.db_path = os.path.join(self.project_dir, "contents.db")
        self.html_path = os.path.join(self.project_dir, "html")
        self.htmlind_path = os.path.join(self.html_path, "index.html")
        self.image_path = os.path.join(self.project_dir, "image")

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed + 1)

        self.display = result_display.HTMLDisplay(
            self.db_path, self.name, self.html_path, self.image_path)

        if doc is None:
            doc = ''
        doc = f"{self.name}\n{doc}"
        
        steps_storage = actions_looper.PicklingStepsStorage(
            os.path.join(self.project_dir, "steps"))

        self.actions_looper = actions_looper.ActionsLooper(steps_storage, doc=doc)
    
    def as_action(self, *args, **kwargs) -> Any:
        """delegates to `actions_looper.ActionsLooper.action`"""
        return self.actions_looper.as_action(*args, **kwargs)

    def add_available(self, *args, **kwargs) -> bool:
        """delegates to `actions_looper.ActionsLooper.add_available`"""
        return self.actions_looper.add_available(*args, **kwargs)
    
    def add_available_from_module(self, *args, **kwargs) -> None:
        """delegates to `actions_looper.ActionsLooper.add_available_from_module`
        """
        return self.actions_looper.add_available_from_module(*args, **kwargs)
    
    def add_available_from_main_module(self, *args, **kwargs) -> None:
        """delegates to
        `actions_looper.ActionsLooper.add_available_from_main_module`
        """
        return self.actions_looper.add_available_from_main_module(*args, **kwargs)

    
    def actions_loop(self, *args, **kwargs) -> None:
        """delegates to `actions_looper.ActionsLooper.actions_loop`"""
        return self.actions_looper.actions_loop(*args, **kwargs)

    
    @property
    def curr_action(self) -> str:
        """delegates to `actions_looper.ActionsLooper.curr_action`"""
        return self.actions_looper.curr_action
        
    @property
    def curr_step(self) -> int:
        """delegates to `actions_looper.ActionsLooper.curr_step`"""
        return self.actions_looper.curr_step
        
    @property
    def curr_step_method(self) -> str:
        """delegates to `actions_looper.ActionsLooper.curr_step_method`"""
        return self.actions_looper.curr_step_method
        
    @property
    def curr_astep(self) -> str:
        """delegates to `actions_looper.ActionsLooper.curr_astep`"""
        return self.actions_looper.curr_astep
    
    def execute(self, *args, **kwargs) -> None:
        """delegates to `actions_looper.ActionsLooper.execute`"""
        return self.actions_looper.execute(*args, **kwargs)
    
    def execute_fct_steps(self, *args, **kwargs) -> None:
        """delegates to `actions_looper.ActionsLooper.execute_fct_steps`"""
        return self.actions_looper.execute_fct_steps(*args, **kwargs)
    
    def read_result(self, action: str, step: int) -> Any:
        """delegates to `actions_looper.ActionsLooper.read_result`"""
        return self.actions_looper.read_result(action, step)

    
    def printer(self, *args, **kwargs) -> None:
        """delegates to `result_display.HTMLDisplay.printer`"""
        return self.display.printer(*args, **kwargs)

    def print(self, *args, **kwargs) -> None:
        """delegates to `result_display.HTMLDisplay.print`"""
        return self.display.print(*args, **kwargs)
        
    def savefig(self, *args, **kwargs) -> None:
        """delegates to `result_display.HTMLDisplay.savefig`"""
        return self.display.savefig(*args, **kwargs)
    
    def add_db_entry(self, *args, **kwargs) -> None:
        """delegates to `result_display.HTMLDisplay.add_db_entry`"""
        return self.display.add_db_entry(*args, **kwargs)
        
    def link_text(self, *args, **kwargs) -> str:
        """delegates to `result_display.HTMLDisplay.link_text`"""
        return self.display.link_text(*args, **kwargs)
    
    def print_link_and_return_filepath(self, filename: str, remark: str = '') -> str:
        """Create a filepath for the given filename in the image directory and
        print an HTML link to it.

        Args:
            filename (str): filename (no directory)
            remark (str, optional): remark to print before the link text
        Returns:
            str: filepath in the image directory
        """
        filepath = os.path.join(self.image_path, filename)
        link_text = self.display.link_text(filepath, link_text=filename)
        print(f'{remark}{link_text}')
        return filepath

    def get_analysis_pdf_filepath(self, model_name: str, iteration: int =-1) -> str:
        """Filepath to save a Cyclic Boosting analysis PDF in.
        
        Args:
            model_name (str): model name
            iteration (int): Cyclic Boosting iteration
        Returns:
            str: filepath in the image directory
        """
        if iteration == -1:
            suffix = ""
        else:
            suffix = f"_{iteration}"
        return os.path.join(self.image_path, f"{model_name}{suffix}.pdf")
    
    def get_keys(self) -> List[str]:
        """delegates to `result_display.HTMLDisplay.get_keys`"""
        return self.display.get_keys()

    def del_keys(self, keys: Union[str, List[str]]) -> None:
        """delegates to `result_display.HTMLDisplay.del_keys`"""
        return self.display.del_keys(keys)

    def del_keys_like(self, regex: str) -> None:
        """delegates to `result_display.HTMLDisplay.del_keys_like`"""
        return self.display.del_keys_like(regex)

    def get_findings(self) -> List[Any]:
        """delegates to `result_display.HTMLDisplay.db_findings"""
        return self.display.get_findings()


@contextlib.contextmanager
def temp_project() -> Generator[Manager]:
    """Context manager offering a temporary project

    For example for testing purposes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Manager(tmpdir)
