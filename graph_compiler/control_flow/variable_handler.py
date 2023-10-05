class VariableHandler:
    """
    Reads and writes to cells on behalf of the function.
    This allows the reparsed function to share closures with the original.
    """

    def __init__(self, glob, clos):
        self.globals = glob  # reference to global namespace
        self.closures = clos  # dictionary of cell objects

    def get_var(self, name, default):
        if name in default:
            return default[name]
        if name in self.globals:
            return self.globals[name]
        if name in self.closures:
            return self.closures[name].cell_contents
        return None

    def set_var(self, name, val):
        if name in self.globals:
            self.globals[name] = val
        elif name in self.closures:
            self.closures[name].cell_contents = val
        else:
            raise VariableHandlingException(
                "VariableHandler: Tried to assign variable '"
                + name
                + "' for the first time"
            )

    def aug_var(self, var, op, value):
        """
        Used for runtime augmentation of nonlocal variables
        ToDo: refactor this process: e.g. embedding augassign in the tree
        instead of doing it during runtime
        """
        if op == "Add":
            var += value
        elif op == "Sub":
            var -= value
        elif op == "Mult":
            var *= value
        elif op == "Div":
            var /= value
        elif op == "FloorDiv":
            var //= value
        elif op == "Mod":
            var %= value
        elif op == "Pow":
            var **= value
        elif op == "LShift":
            var <<= value
        elif op == "RShift":
            var >>= value
        elif op == "BitOr":
            var |= value
        elif op == "BitXor":
            var ^= value
        elif op == "BitAnd":
            var &= value
        return var

    def store_temp(self, val):
        # may be used for unpacking nonlocal assignment
        self.temp = val

    def get_temp(self):
        return self.temp


class VariableHandlingException(Exception):
    pass
