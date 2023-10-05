#!/bin/bash

cd /ivy/graph-compiler/ivy_repo
python3 -m pip install --user -e .
cd /ivy/graph-compiler

yes | python3 -m pip uninstall torch
yes | python3 -m pip uninstall paddlepaddle
yes | python3 -m pip uninstall jax
yes | python3 -m pip uninstall jaxlib
python3 -m pytest tests/without_frameworks/test_only_tensorflow.py
