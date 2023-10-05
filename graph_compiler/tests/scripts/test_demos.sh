#!/bin/bash

# install pyvis dependency
python3 -m pip install pyvis
python3 -m pip install imageio
python3 -m pip install flax
python3 -m pip install transformers
python3 -m pip install ipython==8.12.0
python3 -m pip install scikit-image
python3 -m pip install tensorflow

# install the local ivy_repo
cd /ivy/graph-compiler/ivy_repo
python3 -m pip install --user -e .

# Now run the tests
cd /ivy/graph-compiler
python3 -m pytest tests/demos/test_demos.py
