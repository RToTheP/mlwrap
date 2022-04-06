from mlwrap.algorithms import AlgorithmBase, get_algorithm
from mlwrap.data.preparation import prepare_data
from mlwrap.dto import DataDetails, MLSettings, TrainingResults


def train(settings: MLSettings) -> TrainingResults:
    data_details: DataDetails = prepare_data(settings)

    algorithm: AlgorithmBase = get_algorithm(settings=settings)