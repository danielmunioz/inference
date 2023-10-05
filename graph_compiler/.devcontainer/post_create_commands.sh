#!/bin/bash

pip3 install black
pip3 install flake8

git submodule update --init --recursive

cd ivy_repo && python3 -m pip install --user -e .