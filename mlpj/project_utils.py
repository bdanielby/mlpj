"""
Project management utilities
"""
from __future__ import print_function, division

import os
import sys
import re
import random

import numpy as np

from mlpj import python_utils as pu
from mlpj import result_display, actions_looper


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
        verbose (int): verbosity of the result display
    """
    def __init__(self, project_path, seed=None, doc=None, verbose=1):
        self.project_dir = os.path.abspath(os.path.expanduser(project_path))
        pu.makedir_unless_exists(self.project_dir)
        self.name = os.path.basename(name)
        
        self.db_path = os.path.join(self.project_dir, "contents.db")
        self.html_path = os.path.join(self.project_dir, "html")
        self.htmlind_path = os.path.join(self.html_path, "index.html")
        self.image_path = os.path.join(self.project_dir, "image")

        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed + 1)

        self.verbose = verbose
        
        self.display = result_display.HTMLDisplay(
            self.db_path, self.name, self.html_path, self.image_path, verbose=verbose)

        if doc is None:
            doc = ''
        doc = f"{self.name}\n{doc}"
        
        steps_storage = actions_looper.PicklingStepsStorage(
            os.path.join(self.project_dir, "steps"))

        self.actions_looper = actions_looper.ActionsLooper(
            steps_storage, doc=self.doc)
    
    def as_curr(self, *args, **kwargs):
        """delegates to `actions_looper.ActionsLooper.curr`"""
        return self.actions_looper.as_curr(*args, **kwargs)
    
    def as_action(self, *args, **kwargs):
        """delegates to `actions_looper.ActionsLooper.action`"""
        return self.actions_looper.action(*args, **kwargs)

    def add_available(self, *args, **kwargs):
        """delegates to `actions_looper.ActionsLooper.add_available`"""
        return self.actions_looper.add_available(*args, **kwargs)
    
    def add_available_from_module(self, *args, **kwargs):
        """delegates to `actions_looper.ActionsLooper.add_available_from_module`
        """
        return self.actions_looper.add_available_from_module(*args, **kwargs)
    
    def add_available_from_main_module(self, *args, **kwargs):
        """delegates to
        `actions_looper.ActionsLooper.add_available_from_main_module`
        """
        return self.actions_looper.add_available_from_main_module(*args, **kwargs)

    
    def actions_loop(self, *args, **kwargs):
        """delegates to `actions_looper.ActionsLooper.actions_loop`"""
        return self.actions_looper.actions_loop(*args, **kwargs)

    
    @property
    def curr_action(self):
        """delegates to `actions_looper.ActionsLooper.curr_action`"""
        return self.actions_looper.curr_action
        
    @property
    def curr_step(self):
        """delegates to `actions_looper.ActionsLooper.curr_step`"""
        return self.actions_looper.curr_step
        
    @property
    def curr_astep(self):
        """delegates to `actions_looper.ActionsLooper.curr_astep`"""
        return self.actions_looper.curr_astep
        
    @property
    def curr_step_method(self):
        """delegates to `actions_looper.ActionsLooper.curr_step_method`"""
        return self.actions_looper.curr_step_method
    
    def execute(self, *args, **kwargs):
        """delegates to `actions_looper.ActionsLooper.execute`"""
        return self.actions_looper.execute(*args, **kwargs)
    
    def execute_fct_steps(self, *args, **kwargs):
        """delegates to `actions_looper.ActionsLooper.execute_fct_steps`"""
        return self.actions_looper.execute_fct_steps(*args, **kwargs)
    
    def read_result(self, action, step):
        """delegates to `actions_looper.ActionsLooper.read_result`"""
        return self.actions_looper.read_result(action, step)

    
    def printer(self, *args, **kwargs):
        """delegates to `result_display.HTMLDisplay.printer`"""
        return self.display.printer(*args, **kwargs)

    def print(self, *args, **kwargs):
        """delegates to `result_display.HTMLDisplay.print`"""
        return self.display.print(*args, **kwargs)
        
    def savefig(self, *args, **kwargs):
        """delegates to `result_display.HTMLDisplay.savefig`"""
        return self.display.savefig(*args, **kwargs)
    
    def link_text(self, *args, **kwargs):
        """delegates to `result_display.HTMLDisplay.link_text`"""
        return self.display.link_text(*args, **kwargs)
    
    
    def get_keys(self):
        """delegates to `result_display.HTMLDisplay.get_keys`"""
        return self.display.get_keys()

    def del_keys(self, keys):
        """delegates to `result_display.HTMLDisplay.del_keys`"""
        return self.display.del_keys(keys)

    def del_keys_like(self, regex):
        """delegates to `result_display.HTMLDisplay.del_keys_like`"""
        return self.display.del_keys_like(regex)

    
    def generate_htmlpage(self, *args, **kwargs):
        """delegates to `result_display.HTMLDisplay.generate_htmlpage`"""
        return self.display.generate_htmlpage(*args, **kwargs)
