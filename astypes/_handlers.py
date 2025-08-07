from __future__ import annotations

import ast
from dataclasses import dataclass, field
from itertools import chain
from logging import getLogger
from typing import Callable, Iterator, TypeVar

import astroid
import typeshed_client

from ._ass import Ass
from ._helpers import (
    conv_node_to_type, find_variable_assignments, get_parent_function, 
    get_parent_scope, get_ret_type_of_fun, infer, is_assignment_before_node,
    is_camel, qname_to_type,
)
from ._type import Type


logger = getLogger(__package__)
Handler = Callable[[astroid.NodeNG], 'Type | None']
T = TypeVar('T', bound=Handler)


@dataclass
class Handlers:
    _registry: list[tuple[type, Handler]] = field(default_factory=list)

    def get_type(self, node: astroid.NodeNG) -> Type | None:
        """Infer type of the given astroid node.

        If the type cannot be inferred, None is returned.
        Keep in mind that a number of assumptions can be made about the code
        in order to infer the type. Use `Type.assumptions` to see them.
        """
        for supported_type, handler in self._registry:
            if isinstance(node, supported_type):
                result = handler(node)
                if result is not None:
                    assert not result.unknown
                    return result
        return None

    def register(self, t: type) -> Callable[[T], T]:
        def callback(handler):
            self._registry.append((t, handler))
            return handler
        return callback


handlers = Handlers()
get_type = handlers.get_type


@handlers.register(astroid.Return)
def _handle_return(node: astroid.Return) -> Type | None:
    return get_type(node.value)


@handlers.register(astroid.Const)
def _handle_const(node: astroid.Const) -> Type | None:
    if node.value is None:
        return Type.new('None')
    return Type.new(type(node.value).__name__)


@handlers.register(astroid.JoinedStr)
def _handle_fstring(node: astroid.JoinedStr) -> Type | None:
    return Type.new('str')


@handlers.register(astroid.List)
def _handle_list(node: astroid.List) -> Type | None:
    subtype = Type.new('')
    for element_node in node.elts:
        element_type = get_type(element_node)
        if element_type is None:
            return Type.new('list')
        subtype = subtype.merge(element_type)
    if subtype.unknown:
        return Type.new('list')
    return Type.new('list', args=[subtype])


@handlers.register(astroid.Tuple)
def _handle_tuple(node: astroid.Tuple) -> Type | None:
    subtypes = []
    for element_node in node.elts:
        element_type = get_type(element_node)
        if element_type is None:
            return Type.new('tuple')
        subtypes.append(element_type)
    if not subtypes:
        return Type.new('tuple')
    return Type.new('tuple', args=subtypes)


@handlers.register(astroid.Dict)
def _handle_dict(node: astroid.Dict) -> Type | None:
    keys_type = Type.new('')
    for key_node, _ in node.items:
        key_type = get_type(key_node)
        if key_type is None:
            key_type = Type.new('')
            break
        keys_type = keys_type.merge(key_type)

    values_type = Type.new('')
    for _, value_node in node.items:
        value_type = get_type(value_node)
        if value_type is None:
            value_type = Type.new('')
            break
        values_type = values_type.merge(value_type)

    if keys_type.unknown and values_type.unknown:
        return Type.new('dict')
    if keys_type.unknown:
        keys_type = Type.new('Any', module='typing')
    if values_type.unknown:
        values_type = Type.new('Any', module='typing')
    return Type.new('dict', args=[keys_type, values_type])

@handlers.register(astroid.Subscript)
def _handle_subscript(node: astroid.Subscript) -> Type | None:
    t = get_type(node.value)
    if t is None:
        return Type.new('None')
    
    # Check if this is a slice (like a[0:3]) or single index (like a[0])
    if isinstance(node.slice, astroid.Slice):
        # For slices, return the container type
        return t
    else:
        # For single index access, return the element type
        if not t._args:
            return None
        element_type = Type.new('')
        element_type = element_type.merge(t._args[0])
        return element_type

@handlers.register(astroid.Set)
def _handle_set(node: astroid.Set) -> Type | None:
    subtype = Type.new('')
    for element_node in node.elts:
        element_type = get_type(element_node)
        if element_type is None:
            return Type.new('set')
        subtype = subtype.merge(element_type)
    assert not subtype.unknown
    return Type.new('set', args=[subtype])


@handlers.register(astroid.UnaryOp)
def _handle_unary_op(node: astroid.UnaryOp) -> Type | None:
    if node.op == 'not':
        return Type.new('bool')
    result = get_type(node.operand)
    if result is not None:
        result = result.add_ass(Ass.NO_UNARY_OVERLOAD)
        return result
    return None


@handlers.register(astroid.BinOp)
def _handle_binary_op(node: astroid.BinOp) -> Type | None:
    assert node.op
    lt = get_type(node.left)
    if lt is None:
        return None
    rt = get_type(node.right)
    if rt is None:
        return None
    if lt.name == rt.name == 'int':
        if node.op == '/':
            return Type.new('float')
        return lt
    if lt.name in ('float', 'int') and rt.name in ('float', 'int'):
        return Type.new('float')
    if lt.name == rt.name:
        return rt
    return None


@handlers.register(astroid.BoolOp)
def _handle_bool_op(node: astroid.BoolOp) -> Type | None:
    assert node.op
    result = Type.new('')
    for subnode in node.values:
        type = get_type(subnode)
        if type is None:
            return None
        result = result.merge(type)
    return result


@handlers.register(astroid.Compare)
def _handle_compare(node: astroid.Compare) -> Type | None:
    if node.ops[0][0] == 'is':
        return Type.new('bool')
    return Type.new('bool', ass={Ass.NO_COMP_OVERLOAD})


@handlers.register(astroid.ListComp)
def _handle_list_comp(node: astroid.ListComp) -> Type | None:
    return Type.new('list')


@handlers.register(astroid.SetComp)
def _handle_set_comp(node: astroid.SetComp) -> Type | None:
    return Type.new('set')


@handlers.register(astroid.DictComp)
def _handle_dict_comp(node: astroid.DictComp) -> Type | None:
    return Type.new('dict')


@handlers.register(astroid.GeneratorExp)
def _handle_gen_expr(node: astroid.GeneratorExp) -> Type | None:
    return Type.new('Iterator', module='typing')


@handlers.register(astroid.Call)
def _handle_call(node: astroid.Call) -> Type | None:
    if isinstance(node.func, astroid.Attribute):
        result = _get_attr_call_type(node.func)
        if result is not None:
            return result
    if isinstance(node.func, astroid.Name):
        _, symbol_defs = node.func.lookup(node.func.name)
        mod_name = 'builtins'
        if symbol_defs:
            symbol_def = symbol_defs[0]
            if isinstance(symbol_def, astroid.ImportFrom):
                mod_name = symbol_def.modname
        result = get_ret_type_of_fun(mod_name, node.func.name)
        if result is not None:
            return result
        if is_camel(node.func.name):
            return Type.new(node.func.name, ass={Ass.CAMEL_CASE_IS_TYPE})
    return None


@handlers.register(astroid.Name)
def _handle_annotated_attribute(node: astroid.Name) -> Type | None:
    """
    Look in an outer scope of the operation to find a definition
    of the name.
    If the node is a name of an annotated function argument,
    use that annotation.  If the name is defined in a loop body,
    infer the type from the type of the iterator.
    Also handle variable assignments in the current scope.
    """
    # Check for loop variables (pykernel branch feature)
    for parent in node.node_ancestors():
        if isinstance(parent, astroid.For):
            if parent.target.name != node.name:
                continue
            itertype = get_type(parent.iter)
            if itertype._name == "Sequence":
                return itertype._args[0]
            else:
                return None
        if isinstance(parent, astroid.FunctionDef):
            break
    
    func_node = get_parent_function(node)
    
    # Handle function parameters with annotations
    if func_node is not None:
        args = func_node.args
        anns: Iterator[tuple[astroid.AssignName, astroid.NodeNG]] = chain(
            zip(args.args, args.annotations),
            zip(args.posonlyargs, args.posonlyargs_annotations),
            zip(args.kwonlyargs, args.kwonlyargs_annotations),
        )
        for arg, ann in anns:
            if arg.name != node.name:
                continue
            if ann is None:
                continue
            result = conv_node_to_type('__main__', ann)
            if result is None:
                return None
            return result.add_ass(Ass.NO_REDEF)

        if args.vararg is not None and args.vararg == node.name:
            ann = args.varargannotation
            if ann is not None:
                result = conv_node_to_type('__main__', ann)
                if result is not None:
                    return Type.new('tuple', args=[result], ass={Ass.NO_REDEF})
            return Type.new('tuple', ass={Ass.NO_REDEF})

        if args.kwarg is not None and args.kwarg == node.name:
            ann = args.kwargannotation
            if ann is not None:
                result = conv_node_to_type('__main__', ann)
                if result is not None:
                    targs = [Type.new('str'), result]
                    return Type.new('dict', args=targs, ass={Ass.NO_REDEF})
            targs = [Type.new('str'), Type.new('Any', module='typing')]
            return Type.new(name='dict', args=targs, ass={Ass.NO_REDEF})

    # NEW LOGIC: Handle variable assignments
    scope = get_parent_scope(node)
    assignments = find_variable_assignments(node, scope)
    
    if not assignments:
        return None
    
    # Filter assignments that occur before this node
    relevant_assignments = [
        assign for assign in assignments 
        if is_assignment_before_node(assign, node)
    ]
    
    if not relevant_assignments:
        return None
    
    # Collect types from all relevant assignments
    result_type = Type.new('')
    has_annotation = False
    
    for assignment in relevant_assignments:
        # Handle type annotations (AnnAssign) - these take precedence
        if isinstance(assignment, astroid.AnnAssign):
            if assignment.annotation is not None:
                ann_type = conv_node_to_type('__main__', assignment.annotation)
                if ann_type is not None:
                    has_annotation = True
                    result_type = result_type.merge(ann_type)
            # Also consider the assigned value if present
            if assignment.value is not None:
                value_type = get_type(assignment.value)
                if value_type is not None:
                    result_type = result_type.merge(value_type)
        
        # Handle regular assignments (Assign)
        elif isinstance(assignment, astroid.Assign) and assignment.value is not None:
            value_type = get_type(assignment.value)
            if value_type is not None:
                result_type = result_type.merge(value_type)
        
        # Handle augmented assignments (AugAssign)
        elif isinstance(assignment, astroid.AugAssign) and assignment.value is not None:
            value_type = get_type(assignment.value)
            if value_type is not None:
                result_type = result_type.merge(value_type)
    
    if result_type.unknown:
        return None
    
    # Add appropriate assumptions
    if has_annotation:
        result_type = result_type.add_ass(Ass.NO_REDEF)
    else:
        result_type = result_type.add_ass(Ass.ALL_ASSIGNS_SAME)
    
    return result_type


@handlers.register(astroid.NodeNG)
def _handle_infer_any(node: astroid.NodeNG) -> Type | None:
    result = Type.new('')
    for def_node in infer(node):
        if not isinstance(def_node, astroid.Instance):
            result = result.add_ass(Ass.ALL_ASSIGNS_SAME)
            continue
        type = qname_to_type(def_node.pytype())
        if type is None:
            result = result.add_ass(Ass.ALL_ASSIGNS_SAME)
            continue
        result = result.merge(type)
    if result.name in ('', 'None'):
        return None
    return result


@handlers.register(astroid.Call)
def _handle_call_infer(node: astroid.Call) -> Type | None:
    for def_node in infer(node.func):
        if not isinstance(def_node, astroid.FunctionDef):
            continue
        mod_name, _, fun_name = def_node.qname().rpartition('.')
        ret_type = conv_node_to_type(mod_name, def_node.returns)
        if ret_type is not None:
            return ret_type
        ret_type = get_ret_type_of_fun(mod_name, fun_name)
        if ret_type is not None:
            return ret_type
    return None


def _get_attr_call_type(node: astroid.Attribute) -> Type | None:
    expr_type = get_type(node.expr)
    if expr_type is None:
        logger.debug('cannot get type of the left side of attribute')
        return None
    module = typeshed_client.get_stub_names('builtins')
    assert module is not None
    try:
        child_nodes = module[expr_type.name].child_nodes
        assert child_nodes is not None
        method_def = child_nodes[node.attrname]
    except KeyError:
        logger.debug('not a built-in function')
        return None
    if not isinstance(method_def.ast, ast.FunctionDef):
        logger.debug('resolved call target of attr is not a function')
        return None
    ret_node = method_def.ast.returns
    return conv_node_to_type('builtins', ret_node)
