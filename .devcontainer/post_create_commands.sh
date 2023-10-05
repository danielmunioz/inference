#!/bin/bash

git submodule update --init --recursive

python3 -m pip install --user -e ivy

git config --global --add safe.directory /workspaces/inference