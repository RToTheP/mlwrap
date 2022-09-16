import pandas as pd

from mlwrap import pipeline, preparation, scores
from mlwrap.config import MLConfig, TrainingResults
from mlwrap.enums import ProblemType


def train(config: MLConfig, df: pd.DataFrame) -> TrainingResults:
    # split the data

    X_train, X_test, y_train, y_test = preparation.split_data(
        df,
        model_feature_id=config.model_feature_id,
        train_size=config.train_test_split,
        shuffle=config.shuffle_before_splitting,
        problem_type=config.problem_type,
    )

    # pipeline

    model = pipeline.get_pipeline(config, X_train, y_train)

    model.fit(X_train, y_train)

    # scores
    y_pred = model.predict(X_test)
    y_prob = None
    if config.problem_type == ProblemType.Classification:
        y_prob = model.predict_proba(X_test)
    scores_ = scores.calculate_scores(
        config=config,
        model=model,
        total_row_count=df.shape[0],
        y=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )

    scores.print_scores(scores_)

    explanations = None
    if config.explain:
        explanations = model.explain(X_train, global_avg=True)
    scores.print_explanation_result(explanations)

    # then get scores and append them to the results object
    return TrainingResults(scores_, model, explanations)


def clean(df: pd.DataFrame):

    pipeline_ = pipeline.get_cleaning_pipeline()

    df_clean = pipeline_.fit_transform(df)

    return df_clean
