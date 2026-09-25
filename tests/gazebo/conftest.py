import pytest
from support import dockerenv


@pytest.fixture(scope="session")
def built_image():
    if not dockerenv.daemon_running():
        pytest.skip("the Docker daemon is not running")
    return dockerenv.build()
