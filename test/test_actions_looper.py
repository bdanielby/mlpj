"""
Unit tests for `mlpj.actions_looper`.
"""
import collections
from typing import List, Tuple, Any

import pytest

from mlpj import project_utils, actions_looper
from mlpj.project_utils import temp_project


def sorted_items_of_dict(dic: dict) -> List[Tuple[Any, Any]]:
    """Sorted items of a defaultdict

    Args:
        dic (defaultdict): input dictionary
    Returns:
        list: sorted list of items
    """
    res = list(dic.items())
    res.sort()
    return res
    

class ClassExample:
    """Example for an action implemented as a class

    Each step traces its calls in the class attribute `interim_results`.
    """
    interim_results = collections.defaultdict(list)

    def step_r0_init(self) -> str:
        self.interim_results[0].append("0")
        return "init"

    def step_1_next(self, res0: str) -> None:
        self.interim_results[1].append(f"1:{res0}")

    def step_r5(self, res0: str) -> str:
        self.interim_results[5].append(f"5:{res0}")
        return "r2"

    def step_10(self, res0: str, res2: str) -> None:
        self.interim_results[10].append(f"10:{res0},{res2}")

    @classmethod
    def reset(cls) -> None:
        cls.interim_results.clear()

    @classmethod
    def contents(cls) -> List[Tuple[int, List[str]]]:
        return sorted_items_of_dict(cls.interim_results)
    

def test_execute_class_action() -> None:
    with temp_project() as pj:
        pj.add_available(ClassExample)
        ClassExample.reset()

        assert ClassExample.contents() == []
        
        # Previous result producing steps must be called first.
        with pytest.raises(
                FileNotFoundError,
                match=r'No such file or directory.*ClassExample_step0'
        ):
            pj.execute("ClassExample 10")

        pj.execute("ClassExample ..1")
        assert ClassExample.contents() == [(0, ['0']), (1, ['1:init'])]

        with pytest.raises(ValueError, match=r'no step 2 found'):
            pj.execute("ClassExample 2")
        
        pj.execute("ClassExample 5..")
        assert ClassExample.contents() == [
            (0, ['0']), (1, ['1:init']), (5, ['5:init']), (10, ['10:init,r2'])
        ]

        pj.execute("ClassExample 1:10")
        assert ClassExample.contents() == [
            (0, ['0']), (1, ['1:init', '1:init']), (5, ['5:init', '5:init']),
            (10, ['10:init,r2'])
        ]

        with pytest.raises(ValueError,
                           match=r'no match found for action pattern: ClsEx'
        ):
            pj.execute("%ClsEx")

        pj.execute("%Ex.*pl :5")
        assert ClassExample.contents() == [
            (0, ['0', '0']), (1, ['1:init', '1:init', '1:init']),
            (5, ['5:init', '5:init']), (10, ['10:init,r2'])
        ]
        
        pj.execute("ClassExample 1..5")
        assert ClassExample.contents() == [
            (0, ['0', '0']), (1, ['1:init', '1:init', '1:init', '1:init']),
            (5, ['5:init', '5:init', '5:init']), (10, ['10:init,r2'])
        ]

        assert pj.read_result("ClassExample", 0) == "init"
        assert pj.read_result("ClassExample", 5) == "r2"


class ObjExample:
    """Example for an action implemented as an object of a class.

    Each step traces its calls in the instance attribute `interim_results`.
    """
    def __init__(self):
        self.interim_results = collections.defaultdict(list)

    def step_r0_init(self) -> str:
        self.interim_results[0].append("0")
        return "init"

    def step_1_next(self, res0: str) -> None:
        self.interim_results[1].append(f"1:{res0}")

    def step_r5(self, res0: str) -> str:
        self.interim_results[5].append(f"5:{res0}")
        return "r2"

    def step_10(self, res0: str, res2: str) -> None:
        self.interim_results[10].append(f"10:{res0},{res2}")

    def contents(self) -> List[Tuple[int, List[str]]]:
        return sorted_items_of_dict(self.interim_results)

    
def test_execute_object_action() -> None:
    with temp_project() as pj:
        obj = ObjExample()
        pj.add_available(obj)

        assert obj.contents() == []
        
        # Previous result producing steps must be called first.
        with pytest.raises(
                FileNotFoundError,
                match=r'No such file or directory.*ObjExample_step0'
        ):
            pj.execute("ObjExample 10")

        pj.execute("ObjExample ..1")
        assert obj.contents() == [(0, ['0']), (1, ['1:init'])]

        with pytest.raises(ValueError, match=r'no step 2 found'):
            pj.execute("ObjExample 2")
        
        pj.execute("ObjExample 5..")
        assert obj.contents() == [
            (0, ['0']), (1, ['1:init']), (5, ['5:init']), (10, ['10:init,r2'])
        ]

        pj.execute("ObjExample 1:10")
        assert obj.contents() == [
            (0, ['0']), (1, ['1:init', '1:init']), (5, ['5:init', '5:init']),
            (10, ['10:init,r2'])
        ]

        with pytest.raises(ValueError,
                           match=r'no match found for action pattern: ClsEx'
        ):
            pj.execute("%ClsEx")

        pj.execute("%Ex.*pl :5")
        assert obj.contents() == [
            (0, ['0', '0']), (1, ['1:init', '1:init', '1:init']),
            (5, ['5:init', '5:init']), (10, ['10:init,r2'])
        ]
        
        pj.execute("ObjExample 1..5")
        assert obj.contents() == [
            (0, ['0', '0']), (1, ['1:init', '1:init', '1:init', '1:init']),
            (5, ['5:init', '5:init', '5:init']), (10, ['10:init,r2'])
        ]

        assert pj.read_result("ObjExample", 0) == "init"
        assert pj.read_result("ObjExample", 5) == "r2"

        
def test_execute_two_actions() -> None:
    with temp_project() as pj:
        obj = ObjExample()
        pj.add_available(obj)

        with pytest.raises(ValueError, match=r'action "ClassExample" not found'):
            pj.execute("ObjExample ..1 ClassExample ..1")
            
        ClassExample.reset()
        pj.add_available(ClassExample)
        
        pj.execute("ObjExample 0 ClassExample ..1")
        assert obj.contents() == [(0, ['0'])]
        assert ClassExample.contents() == [(0, ['0']), (1, ['1:init'])]


def test_execute_fct_action() -> None:
    interim_results = collections.defaultdict(list)

    def contents() -> List[Tuple[int, List[str]]]:
        return sorted_items_of_dict(interim_results)
    
    with temp_project() as pj:
        def fct_example() -> None:
            """Example for an action implemented as a function.
    
            Each step traces its calls in the `defaultdict` `interim_results`.
            """
            def step_r0_init() -> str:
                interim_results[0].append("0")
                return "init"
    
            def step_1_next(res0: str) -> None:
                assert pj.curr_action == "fct_example"
                assert pj.curr_step == 1
                assert pj.curr_step_method == "step_1_next"
                assert pj.curr_astep == "fct_example_step_1_next"
                interim_results[1].append(f"1:{res0}")
    
            def step_r5(res0: str) -> str:
                interim_results[5].append(f"5:{res0}")
                return "r2"
    
            def step_10(res0: str, res2: str) -> None:
                interim_results[10].append(f"10:{res0},{res2}")
    
            pj.execute_fct_steps(locals())
    
        pj.add_available(fct_example)
        assert contents() == []
        
        # Previous result producing steps must be called first.
        with pytest.raises(
                FileNotFoundError,
                match=r'No such file or directory.*fct_example_step0'
        ):
            pj.execute("fct_example 10")

        pj.execute("fct_example ..1")
        assert contents() == [(0, ['0']), (1, ['1:init'])]

        with pytest.raises(ValueError, match=r'no step 2 found'):
            pj.execute("fct_example 2")
        
        pj.execute("fct_example 5..")
        assert contents() == [
            (0, ['0']), (1, ['1:init']), (5, ['5:init']), (10, ['10:init,r2'])
        ]

        pj.execute("fct_example 1:10")
        assert contents() == [
            (0, ['0']), (1, ['1:init', '1:init']), (5, ['5:init', '5:init']),
            (10, ['10:init,r2'])
        ]

        with pytest.raises(ValueError,
                           match=r'no match found for action pattern: ClsEx'
        ):
            pj.execute("%ClsEx")

        pj.execute("%fct.*mpl :5")
        assert contents() == [
            (0, ['0', '0']), (1, ['1:init', '1:init', '1:init']),
            (5, ['5:init', '5:init']), (10, ['10:init,r2'])
        ]
        
        pj.execute("fct_example 1..5")
        assert contents() == [
            (0, ['0', '0']), (1, ['1:init', '1:init', '1:init', '1:init']),
            (5, ['5:init', '5:init', '5:init']), (10, ['10:init,r2'])
        ]

        assert pj.read_result("fct_example", 0) == "init"
        assert pj.read_result("fct_example", 5) == "r2"

        
def test_as_action() -> None:
    interim_results = collections.defaultdict(list)

    def contents() -> List[Tuple[int, List[str]]]:
        return sorted_items_of_dict(interim_results)
    
    with temp_project() as pj:
        @pj.as_action('myaction')
        def fct_example() -> None:
            """Example for an action implemented as a function.
    
            The decorator renames the action to "myaction".
            
            Each step traces its calls in the `defaultdict` `interim_results`.
            """
            def step_r0_init() -> str:
                assert pj.curr_action == "myaction"
                assert pj.curr_step == 0
                assert pj.curr_step_method == "step_r0_init"
                assert pj.curr_astep == "myaction_step_r0_init"
                interim_results[0].append("0")
                return "init"
    
            def step_1_next(res0: str) -> None:
                interim_results[1].append(f"1:{res0}")
    
            pj.execute_fct_steps(locals())
    
        pj.add_available(fct_example)
        pj.execute("myaction ..1")
        assert contents() == [(0, ['0']), (1, ['1:init'])]
        
        pj.execute("myaction 1")
        assert contents() == [(0, ['0']), (1, ['1:init', '1:init'])]
        
        assert pj.read_result("myaction", 0) == "init"


def test_ActionResultProxy() -> None:
    interim_results = collections.defaultdict(list)

    def contents() -> List[Tuple[int, List[str]]]:
        return sorted_items_of_dict(interim_results)
    
    with temp_project() as pj:
        def fct_example() -> None:
            """Example for an action implemented as a function.
    
            Each step traces its calls in the `defaultdict` `interim_results`.
            """
            def step_r0_init() -> actions_looper.ActionResultProxy:
                interim_results[0].append("0")
                return actions_looper.ActionResultProxy("ClassExample", 0)
    
            def step_1_next(res0: str) -> None:
                interim_results[1].append(f"1:{res0}")
    
            pj.execute_fct_steps(locals())
    
        pj.add_available(fct_example)
        assert contents() == []
        
        ClassExample.reset()
        pj.add_available(ClassExample)
        
        pj.execute("ClassExample ..5 fct_example ..1")
        assert ClassExample.contents() == [
            (0, ['0']), (1, ['1:init']), (5, ['5:init'])
        ]
        assert contents() == [(0, ['0']), (1, ['1:init'])]
