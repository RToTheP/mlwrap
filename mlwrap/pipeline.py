from imblearn.pipeline import Pipeline

from mlwrap import encoders, sampling
from mlwrap.algorithms import get_algorithm
from mlwrap.enums import ProblemType
from mlwrap.config import MLConfig
from mlwrap.feature_selection import VarianceThresholdWrapper


class MLWrapPipeline(Pipeline):
    model_feature_encoder = None

    def __init__(
        self,
        steps,
        *,
        memory=None,
        verbose=False,
        problem_type: ProblemType = None,
        model_feature_encoder=None
    ):
        self.problem_type = problem_type
        self.model_feature_encoder = model_feature_encoder
        super().__init__(steps, memory=memory, verbose=verbose)

    def fit(self, X, y=None, **fit_params):
        # y = encoders.to_numpy(y).reshape(-1, 1)
        yt = self.model_feature_encoder.fit_transform(y, None, **fit_params)
        return super().fit(X, yt, **fit_params)

    def predict(self, X, **predict_params):
        yp = super().predict(X, **predict_params)
        ypt = self.model_feature_encoder.inverse_transform(yp)
        return ypt


def get_pipeline(config: MLConfig, X_train, X_test, y_train, y_test):
    """Function to build a model pipeline based on config"""
    steps = []

    # cleaning
    variance_threshold = VarianceThresholdWrapper()
    steps.append(("variance_threshold", variance_threshold))

    # sampling
    resampler = sampling.get_resampler(X_train, config, config.problem_type)
    if resampler is not None:
        steps.append(("resampler", resampler))

    # transformers
    column_transformer = encoders.get_column_transformer(config, X_train)
    model_feature_encoder = encoders.get_model_feature_encoder(config, y_train)      
    steps.append(("column_transformer", column_transformer))

    # model/estimator algorithm
    estimator = get_algorithm(config, X_train, X_test, y_train, y_test)
    steps.append(("estimator", estimator))

    pipeline = MLWrapPipeline(
        steps=steps,
        problem_type=config.problem_type,
        model_feature_encoder=model_feature_encoder,
    )
    return pipeline
