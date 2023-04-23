"""
Execute selected parts of your program based on persisted results of earlier
steps.

The eligible parts of a program are "steps" of "actions".

Classes with methods whose names start with `step_<step number>` or
`step_r<step number>` are recognized as actions and these methods are their
steps. The "r" in the latter kind of method indicates that they produce
results, so their return value must be serialized for use in methods with higher
step numbers.

In the following example, step numbers 0 and 2 produce results which are passed
to their respectively following methods as positional arguments.
```class MyAction:
    def step_r0(self):
        df = simulate_data(...)
        return df

    def step_1(self, df):
        plot(df['x'], df['y'])

    def step_r2(self, df):
        return (df.mean(), df.var())

    def step_4(self, df, means, variances):
        ...
```
`ActionsLooper` enables calling one or more selected steps from the command
line or programmatically, provided the earlier result producing steps have been
called before.

As `step_1` shows, tuple results are treated as individual results in later steps. To
avoid that, return a list instead.

Action choice:
* `<action name>`: exactly this action
* `%<regular expression>`: Find a match among the existing actions. The match
  must be unique.
* `test`: special action to call `pytest` on the module of the script used in
  the command line

Step choice:
* `2`: step number 2 of the given action
* `0 2 3`: steps number 0, 2, 3 of the given action
* `2..`: all steps of the given action with step number at least 2
* `..3`: all steps of the given action with step number up to 3
* `:3`: all steps of the given action with step number below 3
* `2..8`: all steps with step number between 2 and 8 (including 8)
* `2:8`: all steps with step number between 2 and 8 (not including 8)
* `'<step choice 1> <step choice 2>`: executed in the specified order

Combined choice:
* `MyActionB 2 4 MyActionA 5..`: First execute steps 2 and 4 from action
  `MyActionB`, then execute all steps from 5 of action `MyActionA`.

A function can also be an action. Declare its steps as inner functions and call
`execute_fct_steps(locals())` in the last line of the function body.
"""
from __future__ import print_function, division

import os
import sys
import re
import argparse
import collections
import pickle
import inspect
import types

from . import python_utils as pu
from . import plot_utils as pltu


class PicklingStepsStorage(object):
    """Stores the results of steps of your program in pickled form.

    Args:
        storage_dir (str): directory for storing your results
    """
    def __init__(self, storage_dir):
        self.storage_dir = storage_dir
        pu.makedir_unless_exists(self.storage_dir)

    def get_filepath(self, action, step):
        """filepath for the result of the given step of the given action in your
        program

        Args:
            action (str): name of an action in your program
            step (str): name of a step within that action

        Returns:
            str: path to the pickling file for that step of that action
        """
        return os.path.join(self.storage_dir, f"{action}_step{step}")

    def delegate_ifn(self, result):
        """If the result is an ActionResultProxy, look up the action and step
        within it.

        Args:
            result (object): result of a step of an action

        Returns:
            object: the ultimate result of the given step of the given action
        """
        if isinstance(result, ActionResultProxy):
            return self.read(result.action, result.step)
        else:
            return result
        
    def read(self, action, step):
        """Reads the result of the given step of the given action in your
        program

        Args:
            action (str): name of an action in your program
            step (str): name of a step within that action

        Returns:
            object: the ultimate result of the given step of the given action
                (`delegate_ifn` is called internally)
        """
        filepath = self.get_filepath(action, step)
        with open(filepath, 'rb') as fin:
            #return pickle.load(fin)
            result = pickle.load(fin)
        return self.delegate_ifn(result)

    def write(self, action, step, result):
        """Write the result of the given step of the given action in your
        program

        Args:
            action (str): name of an action in your program
            step (str): name of a step within that action
            result (object): result to write

        Returns:
            object: the ultimate result of the given step of the given action
                (`delegate_ifn` is called internally)
        """
        filepath = self.get_filepath(action, step)
        with open(filepath, 'wb') as fout:
            pickle.dump(result, fout, protocol=2)
        return self.delegate_ifn(result)
    
    
class ActionsLooper(object):
    """Manages the execution of selected steps of your program based on
    pickled results of earlier steps
    
    The actions loop is reentrant, i.e. you can call `execute` while it already
    executes an actions loop.

    Instead of an object of this class, a project manager from module
    `project_utils` can be used because it delegates to this class internally.
    
    Args:
        steps_storage (PicklingStepsStorage): object to handle the storage of
            step results
        doc (str, optional): project docstring to be printed in the command
            line help
        with_termseq (bool): whether to color the command line output of actions
            and steps when they are executed
    """
    def __init__(self, steps_storage, doc='', with_termseq=True):
        self.steps_storage = steps_storage
        
        self._available_actions = {}
        self.add_available(TestAction, 'test')
        self.doc = doc
        self.with_termseq = with_termseq

        # In order to make the class reentrant, the following state variables
        #   are implamented as stacks.
        self._curr_action_stack = []
        self._curr_step_stack = []
        self._curr_step_method_stack = []
        self._curr_step_specs_stack = []
        self.actions_level = 0
        
    def as_action(self, name=None):
        """Decorator to add the decorated class or function as an available action.

        Args:
            decorated (class or function): class or function to be decorated
            name (str, optional): if given, supersedes the wrapped function or
                class name
        Returns:
            wrapped class or function
        """
        def action(decorated):
            if name is None:
                name1 = decorated.__name__
            else:
                name1 = name
            self.add_available(decorated, name=name1)
            return decorated
        return action

    def add_available(self, obj, name=None, name_prefix='', tolerate_empty=False):
        """Add an object or a class or function as an available action if it contains
        steps.

        Args:
            obj: object or class or function representing an action
            name (str, optional): If omitted, the result of `_name_of_obj_or_cls`
                is taken.
            name_prefix (str, optional): added as a prefix if given
            tolerate_empty (bool): Unless `True`, an exception is raised if no
                streps are found
        Returns:
            bool: whether an action was added
        """
        if type(obj) is types.FunctionType:
            if name is None:
                name = obj.__name__
            self._available_actions[name] = (obj, None)
            return True

        step_methods = self.step_methods_of_obj_cls_or_locs(obj)
        if step_methods:
            if name is None:
                name = self._name_of_obj_or_cls(obj)
            name = name_prefix + name
            if name in self._available_actions:
                raise ValueError(f"conflict: action name {name} already taken")
            self._available_actions[name] = (obj, step_methods)
            return True
        elif not tolerate_empty:
            raise ValueError(
                f"no step methods found in this object or class: {name}")
        return False

    def add_available_from_module(
            self, module, name_prefix='', pattern_for_funcs=None, tolerate_empty=False):
        """Add all classes with step methods in the module as available actions.

        Args:
            module (python module object): Classes with step methods will be
                searched here.
            name_prefix (str, optional): name prefix for all actions in this module
            pattern_for_funcs (str, optional): a regular expression, is provided,
                functions with names matching that pattern (using `re.search`)
                are included as single-step actions
            tolerate_empty (bool): Unless `True`, an exception is raised if no
                streps are found
        """
        anything_added = False
        for name in dir(module):
            obj = getattr(module, name)
            if inspect.isclass(obj):
                anything_added |= self.add_available(
                    obj, name_prefix=name_prefix, tolerate_empty=True)
            elif pattern_for_funcs is not None and callable(obj):
                if re.search(pattern_for_funcs, name) is not None:
                    anything_added |= self.add_available(obj, obj.__name__)
                
        if not tolerate_empty and not anything_added:
            raise ValueError(
                f"no suitable classes found in module {module.__name__}")

    def add_available_from_main_module(
            self, name_prefix='', pattern_for_funcs=None, tolerate_empty=False):
        """Convenience function for calling `add_available_from_module` for
        the `sys.modules['__main__']`.

        Args:
            name_prefix (str, optional): name prefix for all actions in this module
            pattern_for_funcs (str, optional): a regular expression, is provided,
                functions with names matching that pattern (using `re.search`)
                are included as single-step actions
            tolerate_empty (bool): Unless `True`, an exception is raised if no
                streps are found
        """
        return self.add_available_from_module(
            sys.modules['__main__'], name_prefix=name_prefix,
            pattern_for_funcs=pattern_for_funcs, tolerate_empty=tolerate_empty)

    def actions_loop(self, args=None, add_main=True, pattern_for_funcs=None, tolerate_empty=True
                   ):
        """Execute the actions as specified in the "-a", "--action" args in
        args.

        Args:
            args (list of str, optional): programmatic command line arguments,
                defaulting to `sys.argv[1:]`
            add_main (bool): If yes, call `add_available_from_module` with
                `pattern_for_funcs` and `tolerate_empty`.
            pattern_for_funcs (str, optional): See `add_available_from_module`.
            tolerate_empty (bool): See `add_available_from_module`.
        """
        if args is None:
            args = sys.argv[1:]
            
        if add_main:
            self.add_available_from_main_module(
                tolerate_empty=tolerate_empty,
                pattern_for_funcs=pattern_for_funcs)
            
        parser = self.action_parser()
        self._parse_sys_args()
        self.execute(self.args.actions)

    @property
    def curr_action(self):
        """The currently executed action.

        Returns:
            str: action name
        """
        return self._curr_action_stack[-1]
        
    @property
    def curr_step(self):
        """The currently executed step.

        Returns:
            int: step number
        """
        return self._curr_step_stack[-1]

    @property
    def curr_step_method(self):
        """The currently executed step method.

        Returns:
            str: method name
        """
        return self._curr_step_method_stack[-1]

    @property
    def curr_astep(self):
        """Action and method name of the currently executed step method

        Returns:
            str: `<action>_<method_name>`
        """
        return f"{self.curr_action}_{self.curr_step_method}"

    def execute(self, requested_action_specs):
        """Execute the registered actions and steps selected by the arguments.

        Args:
            requested_action_specs (str or list): either a list of arguments as produced
                by the "-a" command line option or a single string which is then
                split to create such a list
        """
        to_execute = self._translate_requested_actions(requested_action_specs)
        for action, req_steps in to_execute:
            self._curr_action_stack.append(action)
            try:
                obj_or_cls, step_methods = self._available_actions[action]
                if inspect.isclass(obj_or_cls):
                    obj = obj_or_cls()
                else:
                    obj = obj_or_cls

                self._execute1(obj, step_methods, req_steps)
            finally:
                self._curr_action_stack.pop()
                
    def execute_fct_steps(self, locs):
        """Callback to be used from within function actions. It executes the
        requested steps among the steps found in the passed `locals()` dict.
        
        A function can also be an actions. Declare its steps as inner functions
        and call `execute_fct_steps(locals())` in the last line of the function
        body.
        
        Args:
            locs: local variables dict produced by `locals()` from within the
                function body
        """
        step_methods = self.step_methods_of_obj_cls_or_locs(locs)
        req_steps = self._translate_requested_steps(
            self.curr_action, step_methods,
            self._curr_step_specs_stack[-1])
        self._execute_steps_of_action(locs, step_methods, req_steps)

    def read_result(self, action, step):
        """Read the result for the given action and step from the storage.

        Args:
            action (str): name of an action in your program
            step (str): name of a step within that action

        Returns:
            object: the ultimate result of the given step of the given action
        """
        return self.steps_storage.read(action, step)
        
    def action_parser(self):
        """Generates the command line argument parser for the "-a", "--actions"
        arguments.

        This can be overridden in subclasses if this superclass method is
        called.

        Returns:
           `argparse.ArgumentParser`
        """
        parser = argparse.ArgumentParser(
            description=self.doc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '-a', "--actions", type=str, nargs='*', required=True,
            help="List of actions to execute")
        self.argparse = parser
        return parser

    def _cls_of_obj_or_cls(self, obj_or_cls):
        """If the passed object is already a class, return it, otherwise return its
        class.

        Args:
            obj_or_cls: object or class representing an action

        Returns:
            class for the object
        """
        if inspect.isclass(obj_or_cls):
            return obj_or_cls
        else:
            return obj_or_cls.__class__

    def _name_of_obj_or_cls(self, obj_or_cls):
        """Class name of the result of `_cls_of_obj_or_cls`.

        Args:
            obj_or_cls: object or class representing an action

        Returns:
            class name of the object
        """
        return self._cls_of_obj_or_cls(obj_or_cls).__name__

    def _is_locals(self, obj_cls_or_locs):
        """Is the argument is a locals dictionary?

        Args:
            obj_cls_or_locs: object or class or locals dict representing an action
        Returns:
            bool: True if it's a dictionary
        """
        return type(obj_cls_or_locs) is dict

    def _entry_of_obj_cls_or_locs(self, obj_cls_or_locs, name):
        """The step entry for the given name in the argument.

        Args:
            obj_cls_or_locs: object or class or locals dict representing an action
        Returns:
            function or method corresponding to the step name
        """
        if self._is_locals(obj_cls_or_locs):
            return obj_cls_or_locs[name]
        else:
            return getattr(obj_cls_or_locs, name)

    def step_methods_of_obj_cls_or_locs(self, obj_cls_or_locs):
        """The step methods for the argument.
        
        Args:
            obj_cls_or_locs: object or class or locals dict representing an action
        Returns:
            `collections.OrderedDict`: mapping the step number to the step
            function
        """
        step_methods = {}
        rex = re.compile(r'step_r?(\d+).*')

        if self._is_locals(obj_cls_or_locs):
            names = list(obj_cls_or_locs.keys())
        else:
            names = dir(obj_cls_or_locs)
            
        for name in names:
            attr = self._entry_of_obj_cls_or_locs(obj_cls_or_locs, name)
            if not callable(attr):
                continue
            name = attr.__name__
            mo = rex.match(name)
            if mo is not None:
                step_no = int(mo.group(1))
                step_methods[step_no] = name
                
        # Sort the steps numerically.
        step_nos = list(step_methods.keys())
        step_nos.sort()
        step_methods = collections.OrderedDict([
            (step_no, step_methods[step_no]) for step_no in step_nos])
        return step_methods
    
    RE_STEP_NOT_ACTION = r'^(\d+)?(\.\.|:)?(\d+)?$'
    
    def _translate_requested_actions(self, requested_action_specs):
        """Translate the action specifications into a list of actions and requested
        steps.

        Args:
            requested_action_specs: either a string which is then split into
                a list first; or a list of command line action specs or an action

        Returns:
            list of pairs:
            `(<name of requested action>, <list of requested steps>)`
        """
        to_execute = []
        cur_action = None
        cur_step_specs = [[]]
            
        def add_requested_steps():
            if cur_action is not None:
                step_methods = self._available_actions[cur_action][1]
                to_execute.append(
                    (cur_action, self._translate_requested_steps(
                        cur_action, step_methods, cur_step_specs[0])))
                cur_step_specs[0] = []
    
        if pu.isstring(requested_action_specs):
            requested_action_specs = requested_action_specs.split()
        elif not isinstance(requested_action_specs, (tuple, list)):
            cur_action = requested_action_specs.__name__
            add_requested_steps()
            return to_execute
        
        for req_action in requested_action_specs:
            match = re.match(self.RE_STEP_NOT_ACTION, req_action)
            if match is None:
                # action specification encountered
                match = re.match(r'[a-zA-Z_][a-zA-Z0-9_]*|%.*', req_action)
                if match is None:
                    raise ValueError(
                        f'invalid action specification "{req_action}"')
                
                add_requested_steps()
                if req_action.startswith('%'):
                    pattern = req_action[1:]
                    matched_names = [
                        name for name in self._available_actions
                        if re.search(pattern, name)]
                    if len(matched_names) == 0:
                        raise ValueError(
                            f"no match found for action pattern: {pattern}")
                    elif len(matched_names) > 1:
                        raise ValueError(
                            "no unique match found for action pattern: "
                            f"{pattern}")
                    cur_action = matched_names[0]
                else:
                    if req_action not in self._available_actions:
                        raise ValueError(f'action "{req_action}" not found')
                    cur_action = req_action
            else:
                # step specification encountered
                if cur_action is None:
                    raise ValueError(
                        "translate_requested_actions: A step specification "
                        "must be preceded by a an action.")
                cur_step_specs[0].append(req_action)
                
        add_requested_steps()
        return to_execute
            
    def _translate_requested_steps(
            self, cur_action, step_methods, requested_step_specs
    ):
        """Translate step specifications into a list of steps.

        Args:
            cur_action (str): name of curr action
            step_methods (dict): mapping step number to method name
            requested_step_specs: list of integers or integer ranges built with
                ":" or ".."
        Returns:
            list of int: list of step numbers
        """
        if step_methods is None:
            return requested_step_specs
        available_steps = step_methods.keys()

        if not requested_step_specs:
            # No listing of step specifications means: Take all available steps.
            return available_steps

        cur_steps = []
        for req_step in requested_step_specs:
            match = re.match(self.RE_STEP_NOT_ACTION, req_step)
            assert match is not None
            step_start = match.group(1)
            if step_start is None:
                step_start = 0
            else:
                step_start = int(step_start)
            sep = match.group(2)

            if sep is None:
                if step_start not in available_steps:
                    raise ValueError(
                        f"no step {step_start} found in action {cur_action}")
                cur_steps.append(step_start)
            else:
                step_ub = match.group(3)
                steps_to_add = []
                if step_ub is None:
                    for stepno in available_steps:
                        if stepno >= step_start:
                            steps_to_add.append(stepno)
                else:
                    step_ub = int(step_ub)
                    if sep == '..':
                        step_ub += 1
                    for stepno in available_steps:
                        if stepno >= step_start and stepno < step_ub:
                            steps_to_add.append(stepno)
                if not steps_to_add:
                    raise ValueError("no steps found matching the range "
                                     f"{req_action} in action {cur_action}")
                cur_steps.extend(steps_to_add)
        return cur_steps

    def _is_step_with_result(self, step_methods, step):
        """Does the given step method produce a result?

        That is, does its name start with "step_r"?
        
        Returns:
            bool: True for step methods producing results
        """
        return step_methods[step].startswith('step_r')

    def _method_has_params_to_read(self, obj, method_name, step):
        """Has the step method positional or varargs params to read?

        Args:
            obj: object, class or `locals()` dict representing an action
            method_name (str): method name
            step (int): step number
        Returns:
            bool: whether step method has positional params to read
        """
        mthd = self._entry_of_obj_cls_or_locs(obj, method_name)
        argspec = inspect.getfullargspec(mthd)
        n_positional = len(argspec.args)
        if n_positional > 0 and argspec.args[0] == 'self':
            n_positional -= 1
        if argspec.defaults is not None:
            n_positional -= len(argspec.defaults)
        return n_positional > 0 or argspec.varargs is not None
    
    def _steps_to_read(self, obj, step_methods, req_steps):
        """Returns the step numbers of the requested steps that have an
        positional or varargs params to read.

        No need to read any data if the requested methods have no input params
        to read from earlier steps.
        
        Args:
            obj: object, class or `locals()` dict representing an action
            step_methods (dict): mapping step number to method name
            req_steps (sequence of int): requested step numbers
        Returns:
            list of ints: step numbers
        """
        req_steps_with_inputs = [
            step for step in req_steps
            if self._method_has_params_to_read(obj, step_methods[step], step)]
        if not req_steps_with_inputs:
            return []
        
        max_req_step = max(req_steps_with_inputs)
        req_steps = set(req_steps)
        return [
            step for step, method_name in step_methods.items()
            if (step < max_req_step and step not in req_steps
                and self._is_step_with_result(step_methods, step))
        ]

    def _call_step_method(self, obj, method_name, cur_step, data):
        """Call a step method with the required inputs.

        Args:
            obj: object, class or `locals()` dict representing an action
            method_name (str): method name
            cur_step (int): step number
            data (dict): mapping step number to step result
        Returns:
            result of the method call
        """
        mthd = self._entry_of_obj_cls_or_locs(obj, method_name)
        if not self._method_has_params_to_read(obj, method_name, cur_step):
            return mthd()
        
        args = []
        for step, result in data.items():
            if step < cur_step:
                if type(result) is type(()):
                    args.extend(result)
                else:
                    args.append(result)
            else:
                break
        return mthd(*args)
    
    def _execute1(self, obj, step_methods, req_steps):
        """Execute the requested steps for the given action.

        Args:
            obj: object, class or `locals()` dict representing an action
            step_methods (dict): mapping step number to method name
            req_steps (sequence of int): requested step numbers
        """
        if step_methods is None:
            # If there are no step methods, the object must be a
            #   function. Call it. It may call {⠠d execute_fct_steps}
            #   back.
            self._curr_step_specs_stack.append(req_steps)
            try:
                obj()
            finally:
                self._curr_step_specs_stack.pop()
        else:
            self._execute_steps_of_action(obj, step_methods, req_steps)

    def _execute_steps_of_action(self, obj, step_methods, req_steps):
        """Execute the requested steps for the given action which mustn't be a
        function.

        Args:
            obj: object, class representing an action
            step_methods (dict): mapping step number to method name
            req_steps (sequence of int): requested step numbers
        """
        action = self.curr_action
        # Since a requested step can be mentioned multiple times and in any
        #   order, we must first read the required results and then loop
        #   over the requested steps. A loop over all steps using a set of
        #   requested steps wouldn't do.
        data = collections.OrderedDict()
        for step in self._steps_to_read(obj, step_methods, req_steps):
            data[step] = self.steps_storage.read(action, step)

        for req_step in req_steps:
            method_name = step_methods[req_step]
            self._curr_step_stack.append(req_step)
            self._curr_step_method_stack.append(method_name)
            try:
                with pltu.libstyle():
                    print("@" * len(self._curr_action_stack), end=' ')
                    if self.with_termseq:
                        print("{}{}{}.{}{}{}".format(
                            pu.TERMSEQ['cyan['], action, pu.TERMSEQ[']'],
                            pu.TERMSEQ['green['], method_name, pu.TERMSEQ[']']))
                    else:
                        print(f"{action}.{method_name}")
                    result = self._call_step_method(
                        obj, method_name, req_step, data)
                    if self._is_step_with_result(step_methods, req_step):
                        result = self.steps_storage.write(action, req_step, result)
                        data[req_step] = result
            finally:
                self._curr_step_method_stack.pop()
                self._curr_step_stack.pop()
    
    def _parse_sys_args(self, args=None):
        """Call `action_parser`, lets it act on `args` and saves the result in
        `self.args`.

        Args:
            args (list of str, optional): programmatic command line arguments,
                defaulting to `sys.argv[1:]`
        """
        if not hasattr(self, 'argparse'):
            self.action_parser()
        self.args = self.argparse.parse_args(args=args)
        return self


class TestAction(object):
    """Special action to call `pytest` on the module of the script used in
    the command line
    """
    def step_0(self):
        with pltu.libstyle():
            pytest.cmdline.main(args=sys.argv[:1] + ['-s'])


class ActionResultProxy:
    """Special result serving as a proxy for another result saved for a given step
    of a given action.
    
    Use this to avoid saving the same big result multiple time.

    Args:
        action (str): The action to refer to.
        step (str): The step of the above action to refer to.
    """
    def __init__(self, action, step):
        self.action = action
        self.step = step
