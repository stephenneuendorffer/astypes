import astroid
import pytest

from astypes import get_type

@pytest.mark.parametrize('expr, type', [
    # literals
    ('1',       'int'),
    ('1.2',     'float'),
    ('"hi"',    'str'),
    ('f"hi"',   'str'),
    ('b"hi"',   'bytes'),
    ('""',      'str'),
    ('None',    'None'),
    ('True',    'bool'),

    # collection literals
    ('[]',          'list'),
    ('[1]',         'list[int]'),
    ('[1, 2]',      'list[int]'),
    ('[x]',         'list'),
    ('[1, ""]',     'list[int | str]'),
    ('()',          'tuple'),
    ('(1,)',        'tuple[int]'),
    ('(x,)',        'tuple'),
    ('(1, x)',      'tuple'),
    ('(1, 2)',      'tuple[int, int]'),
    ('(1, "")',     'tuple[int, str]'),
    ('{}',          'dict'),
    ('{1:2}',       'dict[int, int]'),
    ('{1:x}',       'dict[int, Any]'),
    ('{1:x,2:y}',   'dict[int, Any]'),
    ('{1:x,"":y}',  'dict[int | str, Any]'),
    ('{x:1,y:2}',   'dict[Any, int]'),
    ('{x:1,y:""}',  'dict[Any, int | str]'),
    ('{1,2}',       'set[int]'),
    ('{1,2}',       'set[int]'),
    ('{1,x}',       'set'),
    ('{x}',         'set'),
    ('{x,y}',       'set'),
    ('{1,""}',      'set[int | str]'),

    # Subscripts
    ('[1][0]',      'int'),
    ('[1,2,3][x]',  'int'),
    ('[1.0][0]',    'float'),

    # collection constructors
    ('list()',      'list'),
    ('list(x)',     'list'),
    ('dict()',      'dict'),
    ('dict(x)',     'dict'),
    ('set()',       'set'),
    ('set(x)',      'set'),
    ('tuple()',     'tuple'),
    ('tuple(x)',    'tuple'),

    # other type constructors
    ('int()',       'int'),
    ('int(x)',      'int'),
    ('str()',       'str'),
    ('str(x)',      'str'),
    ('float()',     'float'),
    ('float(x)',    'float'),

    # math operations
    ('3 + 2',       'int'),
    ('3 * 2',       'int'),
    ('3 + 2.',      'float'),
    ('3. + 2',      'float'),
    ('3 / 2',       'float'),
    ('"a" + "b"',   'str'),

    # binary "bool" operations
    ('3 and 2',     'int'),
    ('3 or 2',      'int'),
    ('3. and 2.',   'float'),
    ('3. or 2.',    'float'),

    # operations with known type
    ('not x',       'bool'),
    ('x is str',    'bool'),

    # operations with assumptions
    ('x in (1, 2, 3)',  'bool'),
    ('x < 10',          'bool'),
    ('~13',             'int'),
    ('+13',             'int'),

    # methods of builtins
    ('"".join(x)',      'str'),
    ('[1,2].count(1)',  'int'),
    ('list(x).copy()',  'list'),
    ('[].copy()',       'list'),
    ('[].__iter__()',   'Iterator'), 
    ('range(5)',        'range'),
    ('range(5)[4]',     'int'),

    # builtin functions
    ('len(x)',          'int'),
    ('oct(20)',         'str'),

    # comprehensions
    ('[x for x in y]',      'list'),
    ('{x for x in y}',      'set'),
    ('{x: y for x in z}',   'dict'),
    ('(x for x in y)',      'Iterator'),

    # misc
    ('Some(x)',             'Some'),
])
def test_expr(expr, type):
    node = astroid.extract_node(f'None\n{expr}')
    t = get_type(node)
    assert t is not None
    assert t.annotation == type


@pytest.mark.parametrize('expr', [
    'min(x)',
    'x',
    'X',
    'WAT',
    'wat()',
    'WAT()',
    '+x',
    'x + y',
    '1 + y',
    'x + 1',
    '"a" + 1',
    'str.wat',
    '"hi".wat',
    'None.hi',
    'None.hi()',
    '"hi".wat()',
    'wat.wat',
    'super().something()',
    'len(x).something()',
    '[].__getitem__(x)',
    'x or y',
    'x and y',
    'x = None; x = b(); x',
    'x[0]',
])
def test_cannot_infer_expr(expr):
    node = astroid.extract_node(expr)
    print(node, get_type(node))
    assert get_type(node) is None


@pytest.mark.parametrize('setup, expr, type', [
    ('import math',                 'math.sin(x)',  'float'),
    ('from math import sin',        'sin(x)',       'float'),
    ('my_list = list',              'my_list(x)',   'list'),
    ('def g(x): return 0',          'g(x)',         'int'),
    # ('def g(x:Sequence[int]): return x[0:3]',      'g(x)',  'int'),
    ('def g(x): return x',          'g(3)',         'int'),
    ('def g(x:int) -> int: return x',      'g(x)',  'int'),
    ('def g(x): \n for i in [1,2]:\n  return i',    'g(x)',         'int'),
    ('class foo:\n def g(x):\n  return x',          'foo()',        'foo'),
    ('class foo:\n def g(x):\n  return x\nclass bar(foo):\n def h():\n  return 0',          'bar()',        'bar'),
    ('x = 13',                      'x',            'int'),
    ('x = 1\nif x:\n  x=True',      'x',            'int | bool'),
    ('from datetime import *',      'date(1,2,3)',  'date'),
    ('import numpy as np',          'np.zeros((4,4), int)',   'ndarray'),
    ('import numpy as np',          'np.array((4,4), int)',   'ndarray'),
    ('import numpy as np',          'np.int32(0)',   'int32')

])
def test_astroid_inference(setup, expr, type):
    stmt = astroid.parse(f'{setup}\n{expr}').body[-1]
    assert isinstance(stmt, astroid.Expr)
    t = get_type(stmt.value)
    assert t is not None
    assert t.annotation == type


@pytest.mark.parametrize('sig, type', [
    ('a: int', 'int'),
    ('b, a: int, c', 'int'),
    ('b: float, a: int, c: float', 'int'),
    ('*, a: int', 'int'),
    ('a: int, /', 'int'),
    ('a: list', 'list'),

    # *args and **kwargs
    ('*a: int', 'tuple[int]'),
    ('*a: garbage', 'tuple'),
    ('*a', 'tuple'),
    ('**a: int', 'dict[str, int]'),
    ('**a: garbage', 'dict[str, Any]'),
    ('**a', 'dict[str, Any]'),

    # parametrized generics
    ('a: list[str]', 'list[str]'),
    ('a: list[garbage]', 'list'),
    ('a: dict[str, int]', 'dict[str, int]'),
    ('a: tuple[str, int, float]', 'tuple[str, int, float]'),
    ('a: tuple[str, garbage]', 'tuple'),
])
def test_infer_type_from_signature(sig, type):
    given = f"""
        def f({sig}):
            return a
    """
    func = astroid.parse(given).body[-1]
    assert isinstance(func, astroid.FunctionDef)
    stmt = func.body[-1]
    assert isinstance(stmt, astroid.Return)
    t = get_type(stmt)
    assert t is not None
    assert t.annotation == type

@pytest.mark.parametrize('sig, body, type', [
    ('a: int', 'return a', 'int'),
    ('a: Sequence[int]', 'return a', 'Sequence[int]'),
    ('a: Sequence[int]', 'return a[0]', 'int'),
    ('a: int, y: int', 
        """
            x = a
            return x""", 'int'),
    ('a: np.int32', 
        """
            return a""", 'int32'),
    ('a: np.int16', 
        """
            return a""", 'int16'),
    ('a: int, y: int', 
        """
            x = y = a
            return x""", 'int'),

    ('a: int, y: int', 
        """
            x, _ = (a, 1.0)
            return x""", 'int'),
    ('a: int, y: int', 
        """
            (x, _) = (a, 1.0)
            return x""", 'int'),
    ('a: int, y: int', 
        """
            x += a
            return x""", 'int'),
    ('a: float, y: int', 
        """
            x = y
            a += 1
            return x""", 'int'),
    ('a: int, y: int', 
        """
            x:float = a
            return x""", 'float'),
    ('a: int, y: int', 
        """
            z:float = a
            return a""", 'int'),
    ('a: int, b: str', 
        """
            if a > 0:
                y = a
            else:
                y = b
            return y""", 'int | str'),
    ('a: int, b: str', 
        """
            while a < 0:
                y = a
                a = a + 1
            else:
                y = b
            return y""", 'int | str'),
    ('a: int, b: str', 
        """
            y = a
            y = b
            return y""", 'int | str'),
    ('a: int, b: str', 
        """
            try:
                y = a
            finally:
                y = b
            return y""", 'int | str'),
    ('a: int, b: str', 
        """
            try:
                y = a
            except Exception:
                y = b
            return y""", 'int | str'),
    ('a: int, b: str', 
        """
            try:
                y = a
            except Exception:
                y = b
            else:
                y = b
            return y""", 'int | str'),

    ('a: int, b: str', 
        """
            y = a
            with foo() as t:
                y = b
            return y""", 'int | str'),
    ('a: Sequence[int]', 'return a[0:3]', 'Sequence[int]'),
    ('a: Sequence[int]', 'return a[0:2,0:3]', 'Sequence[int]'),
    ('a: Sequence[int]', """
            x = a[0,0:3]
            y = a[0,0]
            return y""", 'int'),
    ('a: Sequence[int]', 
        """
            x = a
            return x[0:3]""", 'Sequence[int]'),
    ('a: int', """
            y = np.int32(a)
            return y""",   'int32')

])
def test_infer_body(sig, body, type):
    given = f"""
        import numpy as np
        def f({sig}):
            {body}
    """
    print(given)
    func = astroid.parse(given).body[-1]
    print(func)
    assert isinstance(func, astroid.FunctionDef)
    stmt = func.body[-1]
    assert isinstance(stmt, astroid.Return)

    print(stmt)
    t = get_type(stmt.value)
    print(t, ":", stmt.value)
    assert t is not None
    assert t.annotation == type

@pytest.mark.parametrize('sig, sig2, body, type', [
    #('a: int', 'b: float', 'return a', 'int'),
    ('a: int', 'b: float', 'return b', 'float'),
    ('a: float', 'a: int', 'return a', 'int'),
])
def test_infer_body2(sig, sig2, body, type):
    given = f"""
        def f({sig}):
            def g({sig2}):
                {body}
    """
    print(given)
    func = astroid.parse(given).body[-1]
    print(func)
    assert isinstance(func, astroid.FunctionDef)
    func2 = func.body[-1]
    assert isinstance(func2, astroid.FunctionDef)
    stmt = func2.body[-1]
    assert isinstance(stmt, astroid.Return)

    print(stmt)
    t = get_type(stmt.value)
    print(t, ":", stmt.value)
    assert t is not None
    assert t.annotation == type

@pytest.mark.parametrize('sig', [
    '',
    'b',
    'b: int',
    'a',
    'a: garbage',
    'a: garbage[int]',
])
def test_cannot_infer_type_from_signature(sig):
    given = f"""
        def f({sig}):
            return a
    """
    func = astroid.parse(given).body[-1]
    assert isinstance(func, astroid.FunctionDef)
    stmt = func.body[-1]
    assert isinstance(stmt, astroid.Return)
    t = get_type(stmt)
    assert t is None


@pytest.mark.parametrize('sig, type', [
    ('[1,2,3]', 'int'),
    ('range(5)', 'int'),
])
def test_infer_iterator_type_from_signature(sig, type):
    given = f"""
        for i in {sig}:
            return i
    """
    func = astroid.parse(given).body[-1]
    assert isinstance(func, astroid.For)
    stmt = func.body[-1]
    assert isinstance(stmt, astroid.Return)
    t = get_type(stmt)
    assert t is not None
    assert t.annotation == type

@pytest.mark.parametrize('sig, assign, type', [
    ('', "acc = 0", 'int'),
    ('', "y = np.int32", "int32"),
    ('t: int', "a = np.int32(t)", "int32"),
    ('x:Sequence[np.int32]', "b = x", "Sequence[int32]"),
    ('t2: np.int32', "c = t2", "int32"),
    ('x:Sequence[np.int32]', "d = x[0]", "int32"),
])
def test_assign(sig, assign, type):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    given = f"""
import numpy as np
def f({sig}):
    {assign}
    """
    func = astroid.parse(given).body[-1]
    assert isinstance(func, astroid.FunctionDef)
    stmt = func.body[0]
    v = get_type(stmt.value)
    assert v.annotation == type
