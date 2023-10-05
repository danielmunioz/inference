import os
import subprocess


def upgrade_frameworks():
    subprocess.run(
        (
            f"python -m pip install --upgrade jax jaxlib paddlepaddle \
                torch --no-cache-dir --default-timeout=300"
        ),
        shell=True,
    )
    subprocess.run(
        (
            f"python -m pip install --upgrade tensorflow-cpu \
                tensorflow-probability --no-cache-dir --default-timeout=300"
        ),
        shell=True,
    )
    # numpy is automatically upgraded to the latest supported
    # version, as it is a dependency of jax and tensorflow


def run_multiversion_testing():    
    failed = False
    num_passed = 0
    num_failed = 0
    failed_tests = []
    previous_backend = ""
    previous_backend_no_version = ""
    
    # run tests from tests_to_run
    with open("tests_to_run", "r") as f:
        for line in f:
            if line.startswith("tests/"):
                split_line = line.split(",")
                test = split_line[0]
                backend = split_line[1]
                dependencies = split_line[2:] if len(split_line) > 2 else []
                dependencies_string = " ".join(dependencies)
                backend = backend.strip("\n")
                backend_no_version = backend.split("/")[0]

                # upgrade all frameworks between switching frameworks
                if backend_no_version != previous_backend_no_version:
                    upgrade_frameworks()

                # only install new backend if it's a different framework/version
                if backend != previous_backend:
                    os.system(
                        f"python multiversion_testing/multiversion_framework_directory.py {backend} {dependencies_string}"
                    )
                
                # run the test in docker
                ret = os.system(
                    f"pytest -p no:warnings --tb=short {test} --backend {backend_no_version}"
                )
                
                if ret != 0:
                    num_failed += 1
                    failed = True
                    failed_tests.append(backend + " " + test)
                else:
                    num_passed += 1
                
                previous_backend = backend
                previous_backend_no_version = backend_no_version

    upgrade_frameworks()  # reset to latest versions

    # log failed tests
    print('failed tests:', failed_tests)
    with open("failed_tests", "w") as f:
        for test in failed_tests:
            f.write(test + "\n")
    
    total = num_passed + num_failed
    print(str(num_passed) + "/" + str(total), "passed")
    
    if failed:
        exit(1)
    exit(0)


if __name__ == "__main__":
    run_multiversion_testing()
