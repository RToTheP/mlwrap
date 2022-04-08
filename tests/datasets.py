from sklearn.datasets import load_iris

from mlwrap.config import Feature
from mlwrap.enums import FeatureType


class IrisDataset:
    iris = load_iris(as_frame=True)
    df_X = iris["data"]
    df_y = iris["target"]
    target_names = iris["target_names"]
    target_count = len(iris["target_names"])
    model_feature_id = "target"
    features = [
        *[
            Feature(id=name, feature_type=FeatureType.Continuous)
            for name in iris["feature_names"]
        ],
        Feature(id=model_feature_id, feature_type=FeatureType.Categorical),
    ]
