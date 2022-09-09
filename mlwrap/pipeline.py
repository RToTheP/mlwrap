from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from mlwrap import estimators, sampling
from mlwrap.config import MLConfig
from mlwrap.feature_selection import VarianceThresholdWrapper


def get_pipeline(config: MLConfig, df):
    """Function to build a model pipeline based on config"""
    steps = []

    # cleaning
    variance_threshold = VarianceThresholdWrapper()
    steps.append(("variance_threshold", variance_threshold))

    # sampling
    resampler = sampling.get_resampler(df, config, config.problem_type)
    if resampler is not None:
        steps.append(("resampler", resampler))

    # transformers
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                make_column_selector(dtype_exclude="category"),
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include="category"),
            ),
        ]
    )
    steps.append(("preprocessor", preprocessor))

    # model/estimator algorithm
    estimator = estimators.get_estimator(config=config)
    steps.append(("estimator", estimator))

    pipeline = Pipeline(steps=steps)
    return pipeline
