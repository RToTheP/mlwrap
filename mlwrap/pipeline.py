from feature_engine.imputation import CategoricalImputer
from feature_engine.selection import DropConstantFeatures
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer

from mlwrap import algorithms, encoders, explainers, sampling
from mlwrap.config import MLConfig


class MLWrapPipeline(Pipeline):
    model_feature_encoder = None

    def __init__(
        self,
        steps,
        *,
        memory=None,
        verbose=False,
        config: MLConfig = None,
        model_feature_encoder=None,
        explainer=None
    ):
        self.config = config
        self.model_feature_encoder = model_feature_encoder
        self.explainer = explainer
        self.explanations_ = None
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        yt = self.model_feature_encoder.fit_transform(y, None, **fit_params)

        super().fit(X, yt, **fit_params)

        if self.explainer is not None:
            self.explanations_ = self.explainer.fit(X, yt)

        return self

    def predict(self, X, **predict_params):
        yp = super().predict(X, **predict_params)
        ypt = self.model_feature_encoder.inverse_transform(yp)
        return ypt


def get_pipeline(config: MLConfig, X_train, X_test, y_train, y_test):
    """Function to build a model pipeline based on config"""
    steps = []

    # sampling
    resampler = sampling.get_resampler(X_train, config, config.problem_type)
    if resampler is not None:
        steps.append(("resampler", resampler))

    # transformers
    column_transformer = encoders.get_column_transformer(config, X_train)
    model_feature_encoder = encoders.get_model_feature_encoder(config, y_train)
    steps.append(("column_transformer", column_transformer))

    # model/estimator algorithm
    algorithm = algorithms.get_algorithm(config, X_train, X_test, y_train, y_test)
    steps.append(("algorithm", algorithm))

    # explainer
    explainer = None
    if config.explain:
        explainer = explainers.get_explainer(config, algorithm, column_transformer)

    pipeline = MLWrapPipeline(
        steps=steps,
        config=config,
        model_feature_encoder=model_feature_encoder,
        explainer=explainer,
    )
    return pipeline

def get_cleaning_pipeline():
    steps = []

    # transformers to clean feature values
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = CategoricalImputer()
    categorical_columns = ["category", "object"]
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude=categorical_columns)),
            ("cat", categorical_transformer, make_column_selector(dtype_include=categorical_columns)),
        ]
    )
    steps.append(("column_transformer", column_transformer))

    # remove redundant features
    drop_constant_features = DropConstantFeatures()
    steps.append(("drop_constant_features", drop_constant_features))

    return Pipeline(steps=steps)
