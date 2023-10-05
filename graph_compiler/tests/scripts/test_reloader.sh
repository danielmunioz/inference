#!/bin/bash

# install pyvis dependency
python3 -m pip install flax
python3 -m pip install transformers
python3 -m pip install kornia
python3 -m pip install tensorflow

# install the local ivy_repo
cd /ivy/graph-compiler/ivy_repo
python3 -m pip install --user -e .

# Now run the tests
cd /ivy/graph-compiler
python3 -m pytest tests/reloading/test_reloader.py
