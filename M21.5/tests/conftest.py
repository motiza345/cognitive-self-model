import sys
from pathlib import Path

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


@pytest.fixture(scope="session")
def package_dir() -> Path:
    return PACKAGE


@pytest.fixture(scope="session")
def hparams():
    config = yaml.safe_load((PACKAGE / "configs" / "m21_5_v1.yaml").read_text(encoding="utf-8"))
    return config["implementation_choices"]


@pytest.fixture(scope="session")
def ground_truth():
    from environment.env import load_ground_truth

    return load_ground_truth()
