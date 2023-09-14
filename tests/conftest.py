import pytest

from src.dataset import LSCPlantsDataset
from src.metrics import *

PATH_TO_DATA = '../CVPPPSegmData/data'
PATH_TO_SPLIT = '../CVPPPSegmData/split.csv'


def pytest_configure(config):
    pytest.dataset = LSCPlantsDataset(PATH_TO_DATA, PATH_TO_SPLIT).get_train()
    pytest.metrics = (DiffFgDICE, DiffFgMSE)


@pytest.fixture(scope='function')
def small_sample():
    return pytest.dataset[:10]
