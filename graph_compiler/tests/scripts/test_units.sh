#!/bin/bash

# install the local ivy_repo
cd /ivy/graph-compiler/ivy_repo
python3 -m pip install --user -e .

# Now run the tests
cd /ivy/graph-compiler
python3 -m pytest tests/unit/unit_tests.py
