import pytest

from tests import datasets


@pytest.fixture
def iris() -> datasets.IrisDataset:
    return datasets.IrisDataset()


@pytest.fixture
def diabetes() -> datasets.DiabetesDataset:
    return datasets.DiabetesDataset()

@pytest.fixture
def titanic() -> datasets.TitanicDataset():
    return datasets.TitanicDataset()
