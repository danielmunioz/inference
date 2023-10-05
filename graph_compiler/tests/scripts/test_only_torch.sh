#!/bin/bash

cd /ivy/graph-compiler/ivy_repo
python3 -m pip install --user -e .
cd /ivy/graph-compiler

yes | python3 -m pip uninstall tensorflow-cpu
yes | python3 -m pip uninstall jax
yes | python3 -m pip uninstall jaxlib
yes | python3 -m pip uninstall paddlepaddle
python3 -m pytest tests/without_frameworks/test_only_torch.py