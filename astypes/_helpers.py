from __future__ import annotations

import ast
from logging import getLogger

import astroid
import typeshed_client

from ._ass import Ass
from ._type import Type


logger = getLogger(__package__)


def infer(node: astroid.NodeNG) -> list:
    try:
        return list(node.infer())
    except astroid.InferenceError:
        return []


def qname_to_type(qname: str) -> Type:
    if qname.startswith('builtins.'):
        qname = qname.split('.')[-1]
    if qname == 'NoneType':
        qname = 'None'
    if '.' not in qname:
        return Type.new(qname)
    mod_name, _, obj_name = qname.rpartition('.')
    return Type.new(obj_name, module=mod_name)


def is_camel(name: str) -> bool:
    if not name:
        return False
    if not name[0].isupper():
        return False
    if not any(c.islower() for c in name):
        return False
    return True


def get_ret_type_of_fun(
    mod_name: str,
    fun_name: str,
) -> Type | None:
    """For the given module and function name, get return type of the function.
    """
    module = typeshed_client.get_stub_names(mod_name)
    if module is None:
        logger.debug(f'no typeshed stubs for module {mod_name}')
        return None
    fun_def = module.get(fun_name)
    if fun_def is None:
        logger.debug('no typeshed stubs for module')
        return None
    if isinstance(fun_def.ast, ast.FunctionDef):
        ret_node = fun_def.ast.returns
        return conv_node_to_type(mod_name, ret_node)
    if isinstance(fun_def.ast, ast.ClassDef):
        # FIXME: what if there is more than one base?
        type = Type.new(fun_name)
        # for base in fun_def.ast.bases:
        #     basetype = conv_node_to_type(mod_name, base)
        #     if basetype is not None:
        #         print(type, basetype)
        #         type = type.merge(basetype)
        return type
    logger.debug('resolved call target is not a function or class def', fun_def)
    return None


def conv_node_to_type(
    mod_name: str,
    node: ast.AST | astroid.NodeNG | None,
) -> Type | None:
    """Resolve AST node representing a type annotation into a type.
    """
    import builtins
    import typing

    if node is None:
        logger.debug('no return type annotation for called function def')
        return None

    # for generics, try to convert parameters too.
    if isinstance(node, (ast.Subscript, astroid.Subscript)):
        base_type = conv_node_to_type(mod_name, node.value)
        if base_type is None:
            return None
        args: list[Type] = []
        if isinstance(node.slice, (ast.Tuple, astroid.Tuple)):
            for arg_node in node.slice.elts:
                arg_type = conv_node_to_type(mod_name, arg_node)
                if arg_type is None:
                    return base_type
                args.append(arg_type)
        else:
            arg_type = conv_node_to_type(mod_name, node.slice)
            if arg_type is None:
                return base_type
            args.append(arg_type)
        return base_type.add_args(args)

    # for regular name, check if it is a typing primitive or a built-in
    name: str | None = None
    if isinstance(node, ast.Name):
        name = node.id
    if isinstance(node, astroid.Name):
        name = node.name
    if name is not None:
        if hasattr(builtins, name):
            return Type.new(name, ass={Ass.NO_SHADOWING})
        if name in typing.__all__:
            return Type.new(name, module='typing')
        logger.debug(f'cannot resolve {name} into a known type')
        return None

    logger.debug('cannot resolve return AST node into a known type')
    return None


def get_parent_function(node: astroid.NodeNG) -> astroid.FunctionDef | None:
    """Find the node of the function that contains the given node.
    """
    for parent in node.node_ancestors():
        if isinstance(parent, astroid.FunctionDef):
            return parent
    return None


def get_parent_scope(node: astroid.NodeNG) -> astroid.NodeNG:
    """Find the scope that contains the given node (function, class, or module).
    """
    for parent in node.node_ancestors():
        if isinstance(parent, (astroid.FunctionDef, astroid.ClassDef, astroid.Module)):
            return parent
    # Should never happen, but return module as fallback
    return node.root()


def find_variable_assignments(node: astroid.Name, function_scope: astroid.FunctionDef) -> set[astroid.NodeNG]:
    """Find all assignments to a variable that can affect the given node according to Python control flow.
    
    Args:
        node: The Name node we want to find assignments for
        function_scope: The FunctionDef node containing the target node
    
    Returns:
        Set of assignment nodes that precede the target node in execution order
    """
    var_name = node.name
    
    def is_assignment_to_var(stmt) -> astroid.NodeNG | None:
        """Check if a statement assigns to our variable and return the assignment node."""
        if isinstance(stmt, astroid.Assign):
            for target in stmt.targets:
                if isinstance(target, astroid.AssignName) and target.name == var_name:
                    return stmt
        elif isinstance(stmt, astroid.AnnAssign):
            if isinstance(stmt.target, astroid.AssignName) and stmt.target.name == var_name:
                return stmt
        elif isinstance(stmt, astroid.AugAssign):
            if isinstance(stmt.target, astroid.Name) and stmt.target.name == var_name:
                return stmt
        return None
    
    def contains_target_node(tree_node) -> bool:
        """Check if the target node is somewhere in this subtree."""
        if tree_node == node:
            return True
        for child in tree_node.get_children():
            if contains_target_node(child):
                return True
        return False
    
    def traverse_in_execution_order(statements: list[astroid.NodeNG]) -> list[astroid.NodeNG]:
        """Traverse statements in execution order and return assignments that can affect the target."""
        assignments = []
        
        for stmt in statements:
            # If this statement contains our target node, we need to be careful
            if contains_target_node(stmt):
                # Check if the statement itself is an assignment before going deeper
                assignment = is_assignment_to_var(stmt)
                if assignment:
                    assignments.append(assignment)
                
                # Recurse into control structures, but stop when we find the target
                if isinstance(stmt, astroid.If):
                    assignments.extend(traverse_in_execution_order(stmt.body))
                    assignments.extend(traverse_in_execution_order(stmt.orelse))
                elif isinstance(stmt, (astroid.While, astroid.For)):
                    assignments.extend(traverse_in_execution_order(stmt.body))
                    assignments.extend(traverse_in_execution_order(stmt.orelse))
                elif isinstance(stmt, astroid.Try):
                    assignments.extend(traverse_in_execution_order(stmt.body))
                    for handler in stmt.handlers:
                        assignments.extend(traverse_in_execution_order(handler.body))
                    assignments.extend(traverse_in_execution_order(stmt.orelse))
                    assignments.extend(traverse_in_execution_order(stmt.finalbody))
                elif isinstance(stmt, astroid.With):
                    assignments.extend(traverse_in_execution_order(stmt.body))
                
                # Once we've processed the statement containing the target, we're done
                break
            else:
                # This statement doesn't contain the target, so any assignment here can affect it
                assignment = is_assignment_to_var(stmt)
                if assignment:
                    assignments.append(assignment)
                
                # Also check assignments within control structures
                if isinstance(stmt, astroid.If):
                    assignments.extend(traverse_in_execution_order(stmt.body))
                    assignments.extend(traverse_in_execution_order(stmt.orelse))
                elif isinstance(stmt, (astroid.While, astroid.For)):
                    assignments.extend(traverse_in_execution_order(stmt.body))
                    assignments.extend(traverse_in_execution_order(stmt.orelse))
                elif isinstance(stmt, astroid.Try):
                    assignments.extend(traverse_in_execution_order(stmt.body))
                    for handler in stmt.handlers:
                        assignments.extend(traverse_in_execution_order(handler.body))
                    assignments.extend(traverse_in_execution_order(stmt.orelse))
                    assignments.extend(traverse_in_execution_order(stmt.finalbody))
                elif isinstance(stmt, astroid.With):
                    assignments.extend(traverse_in_execution_order(stmt.body))
        
        return assignments
    
    # Start traversal from function body
    assignment_list = traverse_in_execution_order(function_scope.body)
    return set(assignment_list)


def is_assignment_before_node(assignment: astroid.NodeNG, node: astroid.NodeNG) -> bool:
    """Check if an assignment occurs before the given node in execution order.
    
    This is a simplified check based on line numbers.
    """
    return (assignment.lineno or 0) < (node.lineno or 0)
