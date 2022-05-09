from mlwrap.algorithms import AlgorithmBase, get_algorithm
from mlwrap.config import InferenceOutput, MLConfig, TrainingResults
from mlwrap.data.config import DataDetails
from mlwrap.data.preparation import prepare_inference_data, prepare_training_data
from mlwrap.enums import Status
from mlwrap.scores import print_scores, print_explanation_result


def train(config: MLConfig) -> TrainingResults:
    data_details: DataDetails = prepare_training_data(config)

    algorithm: AlgorithmBase = get_algorithm(config=config)

    training_results: TrainingResults = algorithm.fit(data_details)

    print_scores(training_results.scores)

    print_explanation_result(training_results.explanation_result)

    return training_results


def infer(config: MLConfig) -> InferenceOutput:
    data_details = prepare_inference_data(config)
    status = data_details.status

    if status != Status.success:
        return InferenceOutput(
            status=status, cleaning_report=data_details.cleaning_report
        )

    # get the algorithm
    alg = get_algorithm(config=config)
    if not alg.load():
        return

    results: InferenceOutput = alg.infer(data_details)

    return results
