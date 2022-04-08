from mlwrap.algorithms import AlgorithmBase, get_algorithm
from mlwrap.config import  MLConfig, TrainingResults
from mlwrap.data.config import DataDetails
from mlwrap.data.preparation import prepare_training_data



def train(config: MLConfig) -> TrainingResults:
    data_details: DataDetails = prepare_training_data(config)

    algorithm: AlgorithmBase = get_algorithm(config=config)
