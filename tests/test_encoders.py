from mlwrap.config import MLConfig
from mlwrap import encoders
from mlwrap.enums import ProblemType
from tests.datasets import DiabetesDataset, IrisDataset


def test_get_column_transformer_from_features_iris(iris: IrisDataset):
    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        features=iris.features
    )
    column_transformer = encoders.get_column_transformer(config, iris.X)

    transformer_ids = [t[0] for t in column_transformer.transformers]
    for feature in iris.features.values():
        if feature.id == iris.model_feature_id:
            continue
        assert feature.id in transformer_ids

    # difficult to assert on the transformer type so lets just check that the transform works
    Xt = column_transformer.fit_transform(iris.X)
    assert Xt is not None
    assert all(iris.X != Xt)


def test_get_column_transformer_from_dataframe_iris(iris: IrisDataset):
    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
    )
    column_transformer = encoders.get_column_transformer(config, iris.X)

    transformer_ids = [t[0] for t in column_transformer.transformers]
    for feature in iris.features.values():
        if feature.id == iris.model_feature_id:
            continue
        assert feature.id in transformer_ids

    # difficult to assert on the transformer type so lets just check that the transform works
    Xt = column_transformer.fit_transform(iris.X)
    assert Xt is not None
    assert (iris.X != Xt).all(axis=None)

def test_get_model_feature_encoder_from_features_classification(iris: IrisDataset):
    y = iris.y.astype('category')
    y = iris.target_names[y]

    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        features=iris.features
    )
    model_feature_encoder = encoders.get_model_feature_encoder(config, y)

    yt = model_feature_encoder.fit_transform(y)
    assert yt is not None
    assert y != yt

def test_get_model_feature_encoder_from_features_regression(diabetes: DiabetesDataset):

    config = MLConfig(
        model_feature_id=diabetes.model_feature_id,
        problem_type=ProblemType.Classification,
        features=diabetes.features
    )
    model_feature_encoder = encoders.get_model_feature_encoder(config, diabetes.y)

    yt = model_feature_encoder.fit_transform(diabetes.y)
    assert yt is not None
    assert (diabetes.y != yt).all(axis=None)
