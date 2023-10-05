"""
    CFE test helpers
"""

import ast


class FindSCF(ast.NodeVisitor):
    def __init__(self):
        self.has_scf = False
        return super().__init__()

    def visit_If(self, _):
        self.has_scf = True

    def visit_For(self, _):
        self.has_scf = True

    def visit_While(self, _):
        self.has_scf = True


def has_scf(tree):
    find_scf = FindSCF()
    find_scf.visit(tree)
    return find_scf.has_scf
