import pytest
import ivy


@pytest.fixture(autouse=True)
def run_around_tests(dev):
    ivy.unset_backend()
    yield


def pytest_generate_tests(metafunc):
    # device
    raw_value = metafunc.config.getoption("--dev")
    if raw_value == "all":
        devs = ["cpu", "gpu:0", "tpu:0"]
    else:
        devs = raw_value.split(",")

    # create test configs
    configs = list()
    for dev in devs:
        configs.append(dev)
    metafunc.parametrize("dev", configs)
