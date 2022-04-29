from mlwrap.algorithms import AlgorithmBase, get_algorithm
from mlwrap.config import MLConfig, TrainingResults
from mlwrap.data.config import DataDetails
from mlwrap.data.preparation import prepare_training_data
from mlwrap.scores import print_scores, print_explanation_result


def train(config: MLConfig) -> TrainingResults:
    data_details: DataDetails = prepare_training_data(config)

    algorithm: AlgorithmBase = get_algorithm(config=config)

    training_results: TrainingResults = algorithm.fit(data_details)

    print_scores(training_results.scores)

    print_explanation_result(training_results.explanation_result)

    return training_results
