#!/bin/bash
script_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd $script_path

# cythonize with no docstrings
python3 -m cython compiler.pyx --embed -3 --no-docstrings

python3 setup.py build_ext --inplace

# Rename (.so) to compiler.so
mv "_compiler.cpython-38-x86_64-linux-gnu.so" "_compiler.so"

rm "_compiler.c"

# Only execute after compiling with the infuser
file_names=("globals" "tracked_var_proxy" "new_ndarray" "helpers" "source_gen" "param" "frontend_conversion" "graph" "reloader" "wrapping" "compiler" "transpiler" "tracer" "Cacher")

replacement_names=("VII" "VVI" "III" "IIX" "VIV" "IVI" "XIX" "IIV" "XII" "VVV" "XVX" "XXI" "VVX" "XVV")

for ((i=0;i<${#file_names[@]};++i)); do
    mv "${file_names[i]}.so" "${replacement_names[i]}.so"
done
