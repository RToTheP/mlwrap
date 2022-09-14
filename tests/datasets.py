from sklearn.datasets import load_diabetes, load_iris, fetch_openml

from mlwrap.config import Feature
from mlwrap.enums import FeatureType


class IrisDataset:
    dataset = load_iris(as_frame=True)
    X = dataset["data"]
    y = dataset["target"]
    target_names = dataset["target_names"]
    target_count = len(dataset["target_names"])
    model_feature_id = "target"
    features = {
        **{
            name: Feature(id=name, feature_type=FeatureType.Continuous)
            for name in dataset["feature_names"]
        },
        model_feature_id: Feature(
            id=model_feature_id, feature_type=FeatureType.Categorical
        ),
    }


class DiabetesDataset:
    dataset = load_diabetes(as_frame=True)
    X = dataset["data"]
    y = dataset["target"]
    model_feature_id = "target"
    features = {
        **{
            name: Feature(id=name, feature_type=FeatureType.Continuous)
            for name in dataset["feature_names"]
        },
        model_feature_id: Feature(
            id=model_feature_id, feature_type=FeatureType.Continuous
        ),
    }


class TitanicDataset:
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    model_feature_id = "survived"
