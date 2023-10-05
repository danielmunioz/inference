#!/bin/bash

start_time=$(date +%s)

cd graph-compiler

echo "Iteration number: $1"
echo "Installing dependencies..."

# install dependencies
python -m pip install -r ivy_repo/requirements/requirements.txt --no-cache-dir
python -m pip install h5py --no-cache-dir
python -m pip install pytest --no-cache-dir
python -m pip install networkx --no-cache-dir
python -m pip install hypothesis --no-cache-dir
python -m pip install pymongo --no-cache-dir
python -m pip install redis --no-cache-dir
python -m pip install matplotlib --no-cache-dir
python -m pip install opencv-python --no-cache-dir
python -m pip install jax[cpu] --no-cache-dir
python -m pip install jaxlib --no-cache-dir
python -m pip install paddlepaddle --no-cache-dir
python -m pip install tensorflow-cpu --no-cache-dir
python -m pip install tensorflow-addons --no-cache-dir
python -m pip install tensorflow-probability --no-cache-dir
python -m pip install torch --no-cache-dir
python -m pip install scipy --no-cache-dir
python -m pip install dm-haiku --no-cache-dir
python -m pip install pydriller --no-cache-dir
python -m pip install tqdm --no-cache-dir
python -m pip install coverage --no-cache-dir
python -m pip install scikit-learn --no-cache-dir
python -m pip install pandas --no-cache-dir
python -m pip install pyspark --no-cache-dir
python -m pip install autoflake --no-cache-dir
python -m pip install snakeviz --no-cache-dir
python -m pip install imageio --no-cache-dir
python -m pip install diskcache --no-cache-dir
python -m pip install pyvis --no-cache-dir
python -m pip install transformers --no-cache-dir
python -m pip install flax --no-cache-dir
python -m pip install kornia --no-cache-dir
python -m pip install cython --no-cache-dir
python -m pip install ipython==8.12.0 --no-cache-dir

# install the local ivy_repo
python -m pip install --user ivy_repo/. --no-cache-dir

# set environment variables
export IVY_ROOT=".ivy/"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "Collecting multiversion tests..."
# args: iteration number (used as seed), number of tests to run, test directory to sample from
python multiversion_testing/get_multiversion_tests.py $1 250 "tests/"

echo "Running multiversion tests..."
python multiversion_testing/run_multiversion_tests.py
export EXIT_CODE=$?

python -m pip cache purge

end_time=$(date +%s)
duration=$(( end_time - start_time ))
echo "Iteration number: $1"
echo "Multiversion tests took $((duration / 60)) minutes to run."
exit $EXIT_CODE
