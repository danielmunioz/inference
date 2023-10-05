#!/bin/bash

# install pyvis dependency
python3 -m pip install pyvis
python3 -m pip install kornia
python3 -m pip install ipython==8.12.0

# install the local ivy_repo
cd /ivy/graph-compiler/ivy_repo
python3 -m pip install --user -e .

# Now run the tests
cd /ivy/graph-compiler
python3 -m pytest tests/transpilation/test_transpilation.py
