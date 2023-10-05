import ast
import difflib
import inspect
import os
import tokenize
import types
from io import StringIO
from typing import Set, Callable, Tuple, Sequence
from pathlib import Path
from importlib.util import find_spec
from collections import OrderedDict
from collections.abc import MutableSet


# Code from https://stackoverflow.com/a/1653978
class OrderedSet(OrderedDict, MutableSet):

    def __init__(self, x = None):
        if x is not None:
            super().__init__({x:None})
        else:
            super().__init__()

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                 self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))


def trace_obj(obj, args, kwargs, compile_kwargs):
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        obj_call = obj
        source_lines, _ = inspect.getsourcelines(obj_call)
        func_def = source_lines[0].strip()
    elif inspect.isclass(type(obj)):
        if hasattr(obj, "__func__"):
            obj_call = obj.__func__
            source_lines, _ = inspect.getsourcelines(obj)
            func_def = source_lines[0].strip()
        else:
            obj_call = obj.__class__
            func_def = obj.__class__.__name__
    else:
        source_lines, _ = inspect.getsourcelines(obj)
        func_def = source_lines[0].strip()

    code_loc = inspect.getfile(obj_call)
    code_line = inspect.getsourcelines(obj_call)[1]

    args_str = str(args).replace("'", '"')
    kwargs_str = str(kwargs).replace("'", '"')
    compile_kwargs_str = str(compile_kwargs).replace("'", '"')

    return (
        obj_call,
        code_loc,
        code_line,
        func_def,
        args_str,
        kwargs_str,
        compile_kwargs_str,
    )


def retrieve_nested_source_raw(obj, obj_call):
    funcs_source = search_nested_functions(obj_call)

    if os.getenv("IVY_DEBUG_CACHE_TRACE", "False").lower() == "true":
        with open("ivy_temp_cache_trace.txt", "w") as file:
            file.write(funcs_source)

    funcs_source = str(funcs_source).replace("'", '"')
    return funcs_source


def detect_differences(str_func_a, str_func_b):
    difference = difflib.Differ()

    diff_orig = [
        line.strip()
        for line in difference.compare(
            str_func_a.splitlines(keepends=True), str_func_b.splitlines(keepends=True)
        )
    ]

    diff = diff_orig.copy()

    i = 0
    added_lines = []
    removed_lines = []
    changed_lines = []
    while i < len(diff):
        if diff[i].strip():
            if diff[i][0:2] == "? ":
                changed_lines.append(i)
                changed_lines.append(i - 1)
                changed_lines.append(i - 2)
            elif diff[i][0:2] == "- ":
                removed_lines.append(i)
            elif diff[i][0:2] == "+ ":
                added_lines.append(i)
        i += 1

    added_lines = list(set(added_lines) - set(changed_lines))
    removed_lines = list(set(removed_lines) - set(changed_lines))

    return changed_lines, added_lines, removed_lines, diff_orig


def parse_lines(line_numbers, diff_orig):
    line_numbers = [diff_orig[i] for i in line_numbers]
    line_numbers = [line[2:] for line in line_numbers]
    execution_changed = False
    for line in line_numbers:
        line = line.strip()
        if line[0:5] == "print":
            print("print detected, skipping")
        elif line[0:1] == "#":
            print("comment detected, skipping")
        else:
            print("executable line detected: ", line)
            execution_changed = True
    return execution_changed


def parse_all_lines(added_lines, removed_lines, diff_orig):
    added_execution_changed = parse_lines(added_lines, diff_orig)
    removed_execution_changed = parse_lines(removed_lines, diff_orig)
    return added_execution_changed or removed_execution_changed


def fix_indentation(source_str):
    lines = source_str.split("\n")
    if lines[0].startswith(" "):
        # Determine the amount of indentation used
        indentation = len(lines[0]) - len(lines[0].lstrip())
        # Remove the same amount of indentation from every line
        fixed_lines = [line[indentation:] for line in lines]
        return "\n".join(fixed_lines)
    else:
        # Function is already properly indented
        return source_str


def remove_comments_and_docstrings(source):
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        # ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
                    # For example:
                    # def foo():
                    #     "The spaces before this docstring are tokenize.INDENT"
                    #     test = [
                    #         "The spaces before this string do not get a token"
                    #     ]
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out


def get_source_code(obj):
    try:
        source = inspect.getsource(obj)
    except (TypeError, OSError):
        # classes like torch.Tensor don't have source code
        return ""
    source = remove_comments_and_docstrings(source)
    # cleanup source code from single quotes
    # for database storage
    source = fix_indentation(source)
    return source


def get_lib_func_source(func) -> str:
    module_name = func.__module__
    function_name = func.__name__
    module = inspect.getmodule(func)
    if module is not None:
        module_path = module.__file__
        return f"{module_name}.{function_name} at {module_path}\n"


def _get_callable_obj(origin_module, node) -> types.ModuleType:
    # TODO: Handle assignment of function to a variable
    # TODO: Handle the case where the base call id is self or cls
    if isinstance(node, ast.Call):
        return _get_callable_obj(origin_module, node.func)
    if isinstance(node, ast.Attribute):
        return getattr(_get_callable_obj(origin_module, node.value), node.attr)
    elif isinstance(node, ast.Name) and hasattr(origin_module, node.id):
        return getattr(origin_module, node.id)
    else:
        raise AttributeError(f"Can't find the callable object for node: {node}")

def _get_base(node):
    if isinstance(node, ast.Attribute):
        return _get_base(node.value)
    elif isinstance(node, ast.Name):
        return node.id

# TODO:
# 1. Handle function variables inside the class when passed to function
# 2. Handle static and class methods
# 3. Handle class variables
# 4. Handle all kinds of assignments
class CallVistior(ast.NodeVisitor):
    def __init__(self, origin_module: types.ModuleType):
        self.origin_module = origin_module
        self.non_lib_objs = OrderedSet()
        self.lib_objs = OrderedSet()
        self.scope_stack = []
        self.in_class = False
        self.instance_vars = dict()
        self.local_vars = dict()
        self.class_vars = dict()
        self.obj_ref = None
        self.recorded_func_ids = OrderedSet()
        self.class_name = None

    def visit_ClassDef(self, node):
        for decorator in node.decorator_list:
            self._add_called_objs(decorator)
        for base in node.bases:
            self._add_called_objs(base)
        self.scope_stack.append(node)
        self.in_class = True
        self.class_name = node.name 
        self.generic_visit(node)
        self.scope_stack.pop()
        self.in_class = False
        # Reinstate the instace vars
        self.instance_vars = dict()

    def visit_FunctionDef(self, node):
        # instance, class, static, function
        func_type = 'function'
        for decorator in node.decorator_list:
            # Handle classmethod and staticmethod
            if isinstance(decorator, ast.Name) and decorator.id in ['classmethod', 'staticmethod']:
                func_type = decorator.id
            else:
                self._add_called_objs(decorator)

        # check if the function is a intance method
        if self.in_class and func_type != 'staticmethod' and self.scope_stack[-1] == self.class_name:
            # self.instance_vars[node.name] = node
            first_arg = self._get_first_arg(node)
            if func_type == "classmethod":
                self.class_ref = first_arg
            else:
                self.obj_ref = first_arg
        self.scope_stack.append(node)
        for stmt in node.body:
            self.visit(stmt)
        self.local_vars = dict()
        self.scope_stack.pop()
        self.obj_ref = None

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and  self._is_a_obj_ref(node.func):
            if node.func.attr not in self.instance_vars:
                # These are the instance methods and super methods + vars
                return
            assigned_funcs = self.instance_vars[node.func.attr]
            for func in assigned_funcs:
                self._add_called_objs(func)
            return
        if isinstance(node.func, ast.Name):
            if node.func.id in self.local_vars:
                [self._add_called_objs(i) for i in self.local_vars[node.func.id]]

        # handle in a better fashion
        self._add_called_objs(node.func)
        self.generic_visit(node)

    def _add_called_objs(self, func) -> None:
        if isinstance(func, ast.Call):
            return self._add_called_objs(func.func)

        if isinstance(func, ast.Name):
            if func.id in self.recorded_func_ids:
                return
            self.recorded_func_ids.add(func.id)
        elif isinstance(func, ast.Attribute):
            # Need to handle the whole attribute not just the attr
            # if func.attr in self.recorded_func_ids:
            #     return
            # self.recorded_func_ids.add(func.attr)
            pass
        try:
            # This should not be catching python kernel function like len and print
            called_obj = _get_callable_obj(self.origin_module, func)
        except AttributeError:
            return
        if inspect.isfunction(called_obj) or isinstance(called_obj, types.BuiltinFunctionType):
            # TODO: Handle recursion and overflow
            visited_non_lib_objs, visited_lib_objs = get_total_called_functions(called_obj)
            if self.non_lib_objs:
                self.non_lib_objs.update(visited_non_lib_objs)
            else:
                self.non_lib_objs = visited_non_lib_objs
            if self.lib_objs:
                self.lib_objs.update(visited_lib_objs)
            else:
                self.lib_objs = visited_lib_objs
        if inspect.isclass(called_obj) and called_obj.__module__ not in ["builtins", "itertools", "typing"]:
            visited_non_lib_objs, visited_lib_objs = get_total_called_functions(called_obj)
            if self.non_lib_objs:
                self.non_lib_objs.update(visited_non_lib_objs)
            else:
                self.non_lib_objs = visited_non_lib_objs
            if self.lib_objs:
                self.lib_objs.update(visited_lib_objs)
            else:
                self.lib_objs = visited_lib_objs

    def visit_Assign(self, node):
        self._visit_assign(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._visit_assign([node.target], node.value)
        self.generic_visit(node)

    def _visit_assign(self, targets, value) -> None:
        for target in targets:
            if isinstance(target, ast.Attribute):
                
                if self.in_class and isinstance(target.value, ast.Name) and target.value.id == self.obj_ref:
                    values = self.instance_vars.get(target.attr, [])
                    if isinstance(value, ast.Name) and value.id in self.local_vars:
                        # repalce the local variable with contents of local_vars
                        # QUESTION: Cases where the assignment is referential and local change value after assignment
                        value = self.local_vars.get(value.id)
                    value = [value] if not isinstance(value, list) else value 
                    self.instance_vars[target.attr] = values + value
            if isinstance(target, ast.Name):
                values = self.local_vars.get(target.id, [])
                self.local_vars[target.id] = [value] + values
            self.visit(target)

    def _is_a_obj_ref(self, node) -> bool:
        if not self.in_class or self.obj_ref is None:
            return False
        if isinstance(node, ast.Name):
            return node.id == self.obj_ref
        elif isinstance(node, ast.Attribute):
            return self._is_a_obj_ref(node.value) # recusrively call till last
        else:
            return False

    def _get_first_arg(self, node: ast.FunctionDef) -> str:
        # Gets the first argument of a function, which is the instance or class variable
        # should be called when inside a class
        # TODO:Handle vararg (non priority)
        if len(node.args.posonlyargs) == 0: # self is not a positional arg
            if len(node.args.args) == 0: # not passed as normal arg
                return node.args.kwonlyargs[0].arg # should be passed a kwarg (not possible)
            else:
                return node.args.args[0].arg
        else:
            return node.args.posonlyargs[0].arg


def get_total_called_functions(func) -> Tuple[Sequence[Callable], Sequence[Callable]]:
    func_module = inspect.getmodule(func)
    if func_module is None:
        return OrderedSet(), OrderedSet(func)
    module_file_path = os.path.abspath(func_module.__file__) # type: ignore
    if not module_file_path.startswith(os.getcwd()):
        return OrderedSet(), OrderedSet(func)
    python_location =  Path(find_spec("pip").origin).parents[2] # type: ignore
    if module_file_path.startswith(str(python_location)):
        return OrderedSet(), OrderedSet(func)
    if  func_module.__name__.split(".")[0] == "ivy":
        return OrderedSet(), OrderedSet(func)

    rooted_source = get_source_code(func)
    if rooted_source == "": # source code not found
        return OrderedSet(), OrderedSet(func)
    try:
        module_ast = ast.parse(rooted_source)
    except SyntaxError:
        # ast cannot be parsed
        return OrderedSet(), OrderedSet(func)
    visitor = CallVistior(func_module)
    visitor.visit(module_ast)
    if visitor.non_lib_objs:
        return OrderedSet(func), visitor.lib_objs
    visitor.non_lib_objs.update(OrderedSet(func))
    return visitor.non_lib_objs, visitor.lib_objs


def get_funcs_sources(called_funcs) -> str:
    called_funcs_src_list = (get_source_code(func) for func in called_funcs)
    return "".join(called_funcs_src_list)


def get_lib_funcs_sources(called_funcs) -> str:
    called_funcs_src_list = (get_lib_func_source(func) for func in called_funcs)
    return "".join(called_funcs_src_list)


def search_nested_functions(func):
    called_funcs, lib_funcs = get_total_called_functions(func)
    called_funcs_src_dict = get_funcs_sources(called_funcs)
    lib_funcs_src_dict = get_lib_funcs_sources(lib_funcs)
    return called_funcs_src_dict+lib_funcs_src_dict
