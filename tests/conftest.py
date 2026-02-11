import pytest


def pytest_addoption(parser):
    parser.addoption("--show-plots", action="store_true", default=False,
                     help="Show plots when a test fails")
    parser.addoption("--show-plots-always", action="store_true", default=False,
                     help="Always show plots (even on pass)")

@pytest.fixture
def plot_mode(request):
    return {
        "on_fail": request.config.getoption("--show-plots"),
        "always": request.config.getoption("--show-plots-always"),
    }