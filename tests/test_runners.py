import logging
import pandas as pd
import pytest

from mlwrap import io, runners
from mlwrap.config import Feature, MLConfig
from mlwrap.enums import AlgorithmType, FeatureType, ProblemType
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

    features = {
        #    'PassengerId' : Feature(id='PassengerId', feature_type=FeatureType.Categorical),
        "survived": Feature(id="survived", feature_type=FeatureType.Categorical),
        "pclass": Feature(id="pclass", feature_type=FeatureType.Categorical),
        #    'name': Feature(id='name', feature_type=FeatureType.Categorical),
        "sex": Feature(id="sex", feature_type=FeatureType.Categorical),
        "age": Feature(id="age", feature_type=FeatureType.Continuous),
        "sibsp": Feature(id="sibsp", feature_type=FeatureType.Categorical),
        # 'parch': Feature(id='parch', feature_type=FeatureType.Categorical),
        #'Ticket': Feature(id='Ticket', feature_type=FeatureType.Categorical),
        "fare": Feature(id="fare", feature_type=FeatureType.Continuous),
        "embarked": Feature(id="embarked", feature_type=FeatureType.Categorical),
    }
    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        features=features,
        model_feature_id="survived",
    )

    # act
    result = runners.train(config=config, df=df)

    # assert
    assert result.model is not None


def test_e2e_lightgbm_decision_tree_regression(diabetes: DiabetesDataset):
    # arrange
    df = pd.concat([diabetes.X, diabetes.y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        features=diabetes.features,
        model_feature_id=diabetes.model_feature_id,
    )

    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert results.scores['mean_abs_error'] < 70

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


def test_e2e_train_sklearn_linear_classification(iris: IrisDataset):
    # arrange
    # Switch the target column to be string labels to test that scoring is working properly
    df = pd.concat([iris.X, iris.y], axis=1)
    df.target = iris.target_names[df.target]
    df.target = df.target.astype("category")

    config = MLConfig(
        algorithm_type=AlgorithmType.SklearnLinearModel,
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        balance_data_via_resampling=True,
    )

    # training
    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results is not None
    assert results.scores['recall_weighted'] > 0.8

    # save model to disk and then reload
    io.save_model(results.model, "test_data/model.pkl")
    del results
    model = io.load_model("test_data/model.pkl")

    # inference
    # act
    n_inferences = 10
    predictions = model.predict(iris.X.head(n_inferences))

    # assert
    assert len(predictions) == n_inferences
    assert predictions[0] in iris.target_names


def test_e2e_lightgbm_decision_tree_classification(iris: IrisDataset):
    # arrange
    df = pd.concat([iris.X, iris.y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        features=iris.features,
        model_feature_id=iris.model_feature_id,
    )

    # training
    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert results.scores['recall_weighted'] > 0.8

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

    # assert
    assert len(predictions) == n_inferences
    assert iris.target_count == len(probabilities[0])
    assert predictions[0] in iris.y.unique()


def test_e2e_keras_classification(iris):
    # arrange
    df = pd.concat([iris.X, iris.y], axis=1)
    df.target = iris.target_names[df.target]
    df.target = df.target.astype("category")

    config = MLConfig(
        algorithm_type=AlgorithmType.KerasNeuralNetwork,
        features=iris.features,
        model_feature_id=iris.model_feature_id,
    )

    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert results.scores['recall_weighted'] > 0.7

    # save model to disk and then reload
    io.save_model(results.model, "test_data/model.pkl")
    del results
    model = io.load_model("test_data/model.pkl")

    # inference
    # act
    n_inferences = 10
    probabilities = model.predict_proba(iris.X.head(n_inferences))
    predictions = model.predict(iris.X.head(n_inferences))

    # assert
    assert len(predictions) == n_inferences
    assert iris.target_count == len(probabilities[0])
    assert predictions[0] in iris.target_names


def test_e2e_keras_regression(diabetes: DiabetesDataset):
    # arrange
    df = pd.concat([diabetes.X, diabetes.y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.KerasNeuralNetwork,
        features=diabetes.features,
        model_feature_id=diabetes.model_feature_id,
    )

    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert results.scores['mean_abs_error'] < 70

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


def test_xtrain_lightgbm_decision_tree_classification(iris: IrisDataset):
    # arrange
    df = pd.concat([iris.X, iris.y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        features=iris.features,
        model_feature_id=iris.model_feature_id,
        explain=True,
    )

    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert df.shape[0] == results.scores['total_row_count']
    assert results.scores['recall_weighted'] > 0.8
    assert (
        len(config.features) - 1
        == len(results.explanation_result.feature_importances) - 1
    )
    assert any(
        value > 0 for value in results.explanation_result.feature_importances.values()
    )
    assert results.explanation_result.feature_importances["petal length (cm)"] == 1


def test_xtrain_lightgbm_decision_tree_regression(diabetes: DiabetesDataset):
    # arrange
    df = pd.concat([diabetes.X, diabetes.y], axis=1)

    config = MLConfig(
        algorithm_type=AlgorithmType.LightGBMDecisionTree,
        features=diabetes.features,
        model_feature_id=diabetes.model_feature_id,
        explain=True,
    )

    # act
    results = runners.train(config=config, df=df)

    # assert
    assert results.model is not None
    assert df.shape[0] == results.scores['total_row_count']
    assert results.scores['mean_abs_error'] < 70
    assert (
        len(config.features) - 1
        == len(results.explanation_result.feature_importances) - 1
    )
    assert any(
        value > 0 for value in results.explanation_result.feature_importances.values()
    )

    max_feature_importance = max(
        results.explanation_result.feature_importances,
        key=results.explanation_result.feature_importances.get,
    )
    assert max_feature_importance in ["s5", "bmi"]

def test_clean(titanic: TitanicDataset):
    df = pd.concat([titanic.X, titanic.y], axis=1)

    df_clean = runners.clean(df)
    df_clean.columns = df.columns

    initial_missing_values = df.isna().sum().sum()
    final_missing_values = df_clean.isna().sum().sum()
    assert initial_missing_values != final_missing_values
    assert final_missing_values == 0
