import pandas as pd

from mlwrap.algorithms import AlgorithmBase, get_algorithm_old
from mlwrap.config import InferenceOutput, MLConfig, TrainingResults, PipelineResults
from mlwrap.data.config import DataDetails
from mlwrap.data.preparation import (
    prepare_inference_data,
    prepare_training_data,
    split_data_x_y,
)
from mlwrap.enums import ProblemType, Status
from mlwrap import pipeline
from mlwrap.scores import get_pipeline_scores, print_scores, print_explanation_result


def train(config: MLConfig, df: pd.DataFrame) -> TrainingResults:
    data_details: DataDetails = prepare_training_data(config, df)

    algorithm: AlgorithmBase = get_algorithm_old(config=config)

    training_results: TrainingResults = algorithm.fit(data_details)

    print_scores(training_results.scores)

    print_explanation_result(training_results.explanation_result)

    return training_results


def infer(config: MLConfig, df: pd.DataFrame) -> InferenceOutput:
    data_details = prepare_inference_data(config, df)
    status = data_details.status

    if status != Status.success:
        return InferenceOutput(
            status=status, cleaning_report=data_details.cleaning_report
        )

    # get the algorithm
    alg = get_algorithm_old(config=config)
    if not alg.load():
        return

    results: InferenceOutput = alg.infer(data_details)

    return results


def train_pipeline(config: MLConfig, df: pd.DataFrame) -> PipelineResults:
    # process the data - can we use an sklearn pipeline for this?
    # split the data

    X_train, X_test, y_train, y_test = split_data_x_y(
        df,
        model_feature_id=config.model_feature_id,
        train_size=config.train_test_split,
        shuffle=config.shuffle_before_splitting,
        problem_type=config.problem_type,
    )

    # pipeline should:
    # - clean the data
    # - resample the data
    # - fit and transform the data using encoders
    # - fit a model

    model = pipeline.get_pipeline(config, X_train, X_test, y_train, y_test)

    model.fit(X_train, y_train)

    # scores
    scores = get_pipeline_scores(
        config.problem_type,
        model,
        X_test,
        y_test,
    )

    print_scores(scores)

    # then get scores and append them to the results object
    return PipelineResults(scores, model)
