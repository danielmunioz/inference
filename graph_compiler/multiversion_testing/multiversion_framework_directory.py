import subprocess
import sys


def install_pkg_with_dep(backend, dependencies):
    pkg, ver = backend.split("/")[:2]
    install_pkg(pkg + "==" + ver)
    if len(dependencies) > 0:
        install_dependencies(dependencies)


def install_dependencies(dependencies):
    for dep in dependencies:
        subprocess.run(
            (
                f"python -m pip install {dep} --no-cache-dir --default-timeout=100"
            ),
            shell=True,
        )


def install_pkg(pkg):
    root_pkg, version = pkg.split("==")
    
    if pkg.split("==")[0] == "paddle":
        subprocess.run(
            (
                f"python -m pip install paddlepaddle=={version} --no-cache-dir --default-timeout=100"
            ),
            shell=True,
        )
    elif pkg.split("==")[0] == "tensorflow":
        subprocess.run(
            (
                f"python -m pip install tensorflow-cpu=={version} --no-cache-dir --default-timeout=100"
            ),
            shell=True,
        )
    else:
        subprocess.run(
            f"python -m pip install {pkg} --no-cache-dir --default-timeout=100",
            shell=True,
        )


if __name__ == "__main__":
    arg_lis = sys.argv
    backend = arg_lis[1]
    dependencies = arg_lis[2:]
    install_pkg_with_dep(backend, dependencies)
