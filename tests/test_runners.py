import logging
import pandas as pd
import pytest

from mlwrap import io, runners, utils
from mlwrap.config import MLConfig
from mlwrap.enums import AlgorithmType, ProblemType
from tests.datasets import DiabetesDataset, IrisDataset, TitanicDataset

logging.getLogger().setLevel(logging.DEBUG)


def test_train_lightgbm_decision_tree_titanic(titanic: TitanicDataset):
    # arrange
    # columns_to_keep = ['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare' 'cabin', 'embarked', 'boat',
    #    'body', 'home.dest']
    columns_to_keep = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "ticket",
        "fare" "embarked",
        "boat",
        "body",
        "home.dest",
    ]
    df = pd.concat([titanic.X, titanic.y], axis=1)

    df = utils.to_category(df, ["survived", "pclass", "sex", "sibsp", "embarked"])

    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        model_feature_id="survived",
        problem_type=ProblemType.Classification
    )

    # act
    result = runners.train(config=config, df=df)

    # assert
    assert result.model is not None


@pytest.mark.parametrize(
    "algorithm_type",
    [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.KerasNeuralNetwork,
        AlgorithmType.SklearnDecisionTree,
        AlgorithmType.SklearnLinearModel,
    ],
)
def test_e2e_regression(diabetes: DiabetesDataset, algorithm_type: AlgorithmType):
    # arrange
    df = pd.concat([diabetes.X, diabetes.y], axis=1)

    config = MLConfig(
        algorithm_type=algorithm_type,
        model_feature_id=diabetes.model_feature_id,
        problem_type=ProblemType.Regression
    )

    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert results.scores["mean_abs_error"] < 70

    # save model to disk and then reload
    io.save_model(results.model, "test_data/model.pkl")
    del results
    model = io.load_model("test_data/model.pkl")

    # inference
    # act
    n_inferences = 10
    predictions = model.predict(diabetes.X.head(n_inferences))

    # assert
    assert len(predictions) == n_inferences
    assert pytest.approx(predictions[0], 20) == diabetes.y[0]


@pytest.mark.parametrize(
    "algorithm_type",
    [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.KerasNeuralNetwork,
        AlgorithmType.SklearnDecisionTree,
        AlgorithmType.SklearnLinearModel,
    ],
)
def test_e2e_classification(iris: IrisDataset, algorithm_type: AlgorithmType):
    # arrange
    # Switch the target column to be string labels to test that scoring is working properly
    df = pd.concat([iris.X, iris.y], axis=1)
    df.target = iris.target_names[df.target]

    config = MLConfig(
        algorithm_type=algorithm_type,
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        balance_data_via_resampling=True,
    )

    # training
    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results is not None
    assert results.scores["recall_weighted"] > 0.7

    # save model to disk and then reload
    io.save_model(results.model, "test_data/model.pkl")
    del results
    model = io.load_model("test_data/model.pkl")

    # inference
    # act
    n_inferences = 10
    X_test = iris.X.head(n_inferences)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # assert
    assert len(predictions) == n_inferences
    assert iris.target_count == len(probabilities[0])
    assert predictions[0] in iris.target_names


@pytest.mark.parametrize(
    "algorithm_type",
    [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.KerasNeuralNetwork,
        AlgorithmType.SklearnDecisionTree,
        AlgorithmType.SklearnLinearModel,
    ],
)
def test_xe2e_classification(iris: IrisDataset, algorithm_type: AlgorithmType):
    # arrange
    df = pd.concat([iris.X, iris.y], axis=1)

    config = MLConfig(
        problem_type=ProblemType.Classification,
        algorithm_type=algorithm_type,
        model_feature_id=iris.model_feature_id,
        explain=True,
    )

    # training
    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert df.shape[0] == results.scores["total_row_count"]
    assert results.scores["recall_weighted"] > 0.8
    assert (
        iris.X.shape[1]
        == len(results.explanation_result.feature_importances) - 1
    )
    assert any(
        value > 0 for value in results.explanation_result.feature_importances.values()
    )
    max_key = max(results.explanation_result.feature_importances, key=results.explanation_result.feature_importances.get)
    assert max_key in ["petal length (cm)","petal width (cm)"]

    # save model to disk and then reload
    io.save_model(results.model, "test_data/model.pkl")
    del results
    model = io.load_model("test_data/model.pkl")

    # inference
    # act
    n_inferences = 10
    X_test = iris.X.head(n_inferences)
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    explanations = model.explain(X_test, global_avg=False)
    explanation_result = explanations[0]

    # assert
    assert len(predictions) == n_inferences
    assert iris.target_count == len(probabilities[0])
    assert predictions[0] in iris.y.unique()
    assert X_test.shape[1] == len(explanation_result.feature_importances) - 1
    assert any(value > 0 for value in explanation_result.feature_importances.values())
    max_key = max(explanation_result.feature_importances, key=explanation_result.feature_importances.get)
    assert max_key in ["petal length (cm)","petal width (cm)","sepal length (cm)"]


@pytest.mark.parametrize(
    "algorithm_type",
    [
        AlgorithmType.LightGBMDecisionTree,
        AlgorithmType.KerasNeuralNetwork,
        AlgorithmType.SklearnDecisionTree,
        AlgorithmType.SklearnLinearModel,
    ],
)
def test_xe2e_regression(diabetes: DiabetesDataset, algorithm_type: AlgorithmType):
    # arrange
    df = pd.concat([diabetes.X, diabetes.y], axis=1)

    config = MLConfig(
        problem_type=ProblemType.Regression,
        algorithm_type=algorithm_type,
        model_feature_id=diabetes.model_feature_id,
        explain=True,
    )

    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert df.shape[0] == results.scores["total_row_count"]
    assert results.scores["mean_abs_error"] < 70
    assert (
        diabetes.X.shape[1]
        == len(results.explanation_result.feature_importances) - 1
    )
    assert any(
        value > 0 for value in results.explanation_result.feature_importances.values()
    )

    max_feature_importance = max(
        results.explanation_result.feature_importances,
        key=results.explanation_result.feature_importances.get,
    )
    assert max_feature_importance in ["s1", "s3", "s5", "bmi"]

    # save model to disk and then reload
    io.save_model(results.model, "test_data/model.pkl")
    del results
    model = io.load_model("test_data/model.pkl")

    # inference
    # act
    n_inferences = 10
    X_test = diabetes.X.head(n_inferences)
    predictions = model.predict(X_test)
    explanations = model.explain(X_test, global_avg=False)
    explanation_result = explanations[0]

    # assert
    assert len(predictions) == n_inferences
    assert pytest.approx(predictions[0], 20) == diabetes.y[0]
    assert X_test.shape[1] == len(explanation_result.feature_importances) - 1
    assert any(value > 0 for value in explanation_result.feature_importances.values())

    max_feature_importance = max(
        explanation_result.feature_importances,
        key=explanation_result.feature_importances.get,
    )
    assert max_feature_importance in ["s1", "s3", "s5", "bmi"]


def test_clean(titanic: TitanicDataset):
    df = pd.concat([titanic.X, titanic.y], axis=1)

    df_clean = runners.clean(df)
    df_clean.columns = df.columns

    initial_missing_values = df.isna().sum().sum()
    final_missing_values = df_clean.isna().sum().sum()
    assert initial_missing_values != final_missing_values
    assert final_missing_values == 0
