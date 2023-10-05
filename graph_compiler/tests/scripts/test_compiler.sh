#!/bin/bash

# install dependencies
python3 -m pip install pyvis
python3 -m pip install kornia
python3 -m pip install ipython==8.12.0

# install the local ivy_repo
cd /ivy/graph-compiler/ivy_repo
python3 -m pip install --user -e .

# Now run the tests
cd /ivy/graph-compiler
python3 -m pytest tests/compilation/test_compilation.py
