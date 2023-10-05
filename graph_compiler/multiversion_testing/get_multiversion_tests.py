import os
import random
import sys

def get_tests():
    # framework versions to test and their dependencies
    torch_req = [
        # torch versions older than 1.11.0 are no longer available via pip
        "torch/1.11.0",
        "torch/1.12.1",
        "torch/1.13.1",
    ]
    torch_dep = {
        "torch/1.11.0" : ["functorch==0.1.1"],
        "torch/1.12.1" : ["functorch==0.2.1"],
        "torch/1.13.1" : [],
    }
    torch_ignore_tests = {
        "torch/1.10.2" : [
            # functorch is not compatible with 1.10
            "tests/demos/test_demos.py::test_unify_deepmind_perceiver_io",
            "tests/demos/test_demos.py::test_perceiver_io_deepmind",
            "tests/transpilation/test_transpilation.py::test_jax_vmap_transpile",
            "tests/transpilation/test_transpilation.py::test_torch_vmap_transpile",
            # __setitem__ with bool indices is not supported in <= 1.10
            "tests/reloading/test_reloader.py::test_torch_kornia",
            "tests/compilation/test_tracing_caching.py::test_kornia_caching",
        ],
        "torch/1.11.0" : [],
        "torch/1.12.1" : [],
        "torch/1.13.1" : [],
    }
    
    tensorflow_req = [
        # tensorflow versions older than 2.8.0 are no longer available via pip
        "tensorflow/2.8.4",
        "tensorflow/2.9.3",
        "tensorflow/2.10.1",
        "tensorflow/2.11.1",
    ]
    tensorflow_dep = {
        "tensorflow/2.8.4" : ["tensorflow-probability==0.16.0"],
        "tensorflow/2.9.3" : ["tensorflow-probability==0.17.0"],
        "tensorflow/2.10.1" : ["tensorflow-probability==0.18.0"],
        "tensorflow/2.11.1" : ["tensorflow-probability==0.19.0"],
    }
    tensorflow_ignore_tests = {
        "tensorflow/2.8.4" : [],
        "tensorflow/2.9.3" : [],
        "tensorflow/2.10.1" : [],
        "tensorflow/2.11.1" : [],
    }
    
    jax_req = [
        # "jax/0.3.14"
        "jax/0.4.2",
        "jax/0.4.7",
        "jax/0.4.9",
    ]
    jax_dep = {
        "jax/0.3.14" : [
            "jaxlib==0.3.14"
            # "flax==0.5.3",
            # "dm-haiku==0.0.7",
            # "tensorflow-cpu==2.10.1",
            # "tensorflow-probability==0.18.0",
        ],
        "jax/0.4.1" : [
            "flax==0.6.5",
            "jaxlib==0.4.1",
            "jax==0.4.1",  # required due to bug in pip install flax version comparison
        ],
        "jax/0.4.2" : [
            "flax==0.6.5",
            "jaxlib==0.4.2",
            "jax==0.4.2",
            "tensorflow-cpu==2.8.4",
            "tensorflow-probability==0.16.0",
        ],
        "jax/0.4.7" : [
            "flax==0.6.10",
            "jaxlib==0.4.7",
            "jax==0.4.7",
            "tensorflow-cpu==2.12.0",
            "tensorflow-probability==0.20.1",
        ],
        "jax/0.4.9" : [
            "flax==0.6.10",
            "jaxlib==0.4.9",
            "jax==0.4.9",
            "tensorflow-cpu==2.12.0",
            "tensorflow-probability==0.20.1",
        ],
    }
    jax_ignore_tests = {
        "jax/0.3.14" : [],
        "jax/0.4.2" : [],
        "jax/0.4.7" : [],
        "jax/0.4.9" : [],
    }
    
    numpy_req = [
        "numpy/1.20.3",
        "numpy/1.21.6",
        "numpy/1.22.4",
        "numpy/1.23.5",
    ]
    numpy_dep = {
        # these tensorflow dependencies prevent tensorflow from throwing an error relating
        # to numpy compatibility which would otherwise cause the tests to fail
        "numpy/1.20.3" : ["tensorflow-cpu==2.8.4", "tensorflow-probability==0.16.0"],
        "numpy/1.21.6" : ["tensorflow-cpu==2.8.4", "tensorflow-probability==0.16.0"],
        "numpy/1.22.4" : ["tensorflow-cpu==2.12.0", "tensorflow-probability==0.20.0"],
        "numpy/1.23.5" : ["tensorflow-cpu==2.12.0", "tensorflow-probability==0.20.0"],
    }
    numpy_ignore_tests = {
        "numpy/1.20.3" : [],
        "numpy/1.21.6" : [],
        "numpy/1.22.4" : [],
        "numpy/1.23.5" : [],
    }

    paddle_req = [
        "paddle/2.4.2",
    ]
    paddle_dep = {
        "paddle/2.4.2" : [],
    }
    paddle_ignore_tests = {
        "paddle/2.4.2" : [],
    }

    framework_versions = {
        "numpy": numpy_req,
        "torch": torch_req,
        "jax": jax_req,
        "tensorflow": tensorflow_req,
        "paddle": paddle_req,
    }
    
    framework_dependencies = {
        "numpy": numpy_dep,
        "torch": torch_dep,
        "jax": jax_dep,
        "tensorflow": tensorflow_dep,
        "paddle": paddle_dep,
    }

    framework_ignore_tests = {
        "numpy": numpy_ignore_tests,
        "torch": torch_ignore_tests,
        "jax": jax_ignore_tests,
        "tensorflow": tensorflow_ignore_tests,
        "paddle": paddle_ignore_tests,
    }

    # do not use these for the multiversion testing
    tests_to_ignore = [
        "tests/compilation/test_tracing_caching.py",
        "tests/reloading/test_reloading.py",
        "tests/reloading/test_reloading_script.py",
        "tests/without_frameworks/",
        # these are currently failing the checks due to tf import problem
        # so ignoring for now:
        "tests/demos/test_demos.py::test_tensorflow_DeiT",
        "tests/demos/test_demos.py::test_tensorflow_mobile_ViT",
        "tests/reloading/test_reloader.py::test_tf_convnext",
    ]

    # collect arguments, otherwise use default values
    try:
        run_iter = int(sys.argv[1])
    except:
        run_iter = 1
        
    try:
        tests_per_run = int(sys.argv[2])
    except:
        tests_per_run = 150
        
    try:
        tests_path = str(sys.argv[3])
    except:
        tests_path = "tests/"

    # collect test names
    os.system(
        f"pytest --disable-pytest-warnings {tests_path} --my_test_dump true > test_names"
    )
    
    test_names_without_backend = []
    test_names = []

    # get test names from file and remove backend
    with open("test_names") as f:
        for line in f:
            if "ERROR" in line:
                break
            if not line.startswith("tests/"):
                continue
            if any(test in line for test in tests_to_ignore):
                continue

            test_name = line[:-1]
            pos = test_name.find("[")
            if pos != -1:
                test_name = test_name[:pos]
            test_names_without_backend.append(test_name)

    # produce test names with backend and dependencies
    for test_name in test_names_without_backend:
        for backend, backend_versions in framework_versions.items():
            dependencies = framework_dependencies[backend]
            ignore_tests = framework_ignore_tests[backend]

            for backend_version in backend_versions:
                if test_name not in ignore_tests[backend_version]:
                    test_backend = test_name + "," + backend_version
                    version_dep = dependencies[backend_version]
                    
                    for dep in version_dep:  # add framework version dependencies to context
                        test_backend = test_backend + "," + dep
                    
                    test_names.append(test_backend)

    test_names = list(set(test_names))
    test_names.sort()

    # seed random with run_iter for reproducibility
    random.seed(run_iter)
    # sample tests_per_run tests from test_names
    tests_to_run = random.sample(test_names, tests_per_run)

    # sort tests_to_run by backend to minimize pip installations
    tests_to_run.sort(key=lambda x: x.split(',')[1])

    # write tests_to_run to file
    with open("tests_to_run", "w") as f:
        for test in tests_to_run:
            f.write(test + "\n")


if __name__ == "__main__":
    get_tests()
