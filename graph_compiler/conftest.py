# global
import os
import pytest


if "ARRAY_API_TESTS_MODULE" not in os.environ:
    os.environ["ARRAY_API_TESTS_MODULE"] = "ivy.functional.backends.numpy"


def pytest_addoption(parser):
    parser.addoption("--dev", action="store", default="cpu")
    parser.addoption("--backend", action="store", default="jax,numpy,paddle,tensorflow,torch")
    parser.addoption(
        "--my_test_dump",
        action="store",
        default=None,
        help="Print test items in my custom format",
    )
    
def pytest_collection_finish(session):
    if session.config.option.my_test_dump is not None:
        for item in session.items:
            item_path = os.path.relpath(item.path)
            print("{}::{}".format(item_path, item.name))
        pytest.exit("test collection finished")
