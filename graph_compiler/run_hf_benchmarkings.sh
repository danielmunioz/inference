#!/bin/bash

# Number of times to run the script
num_runs=5

# Path to the Python script
python_script="huggingface_benchmarking.py"

for ((i=1; i<=num_runs; i++))
do
    echo "Running script - Iteration: $i"
    python3.10 "$python_script" --demos
done