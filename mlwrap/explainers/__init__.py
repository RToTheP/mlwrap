from mlwrap.config import MLConfig

from mlwrap.explainers.sklearn import (
    SklearnDecisionTreeExplainer,
)


def get_explainer(config: MLConfig, model, column_transformer, background_data):
    model_type = type(model).__name__
    if any(m in model_type for m in ['KerasClassifier', 'KerasRegressor'] ):
        from mlwrap.explainers.shap import GradientSHAP

        return GradientSHAP(
            config=config, model=model, column_transformer=column_transformer, background_data=background_data
        )

    if any(m in model_type for m in ['LGBMClassifier', 'LGBMRegressor'] ):
        from mlwrap.explainers.shap import TreeSHAP

        return TreeSHAP(
            config=config, model=model, column_transformer=column_transformer, background_data=background_data
        )

    

    if any(m in model_type for m in ['LogisticRegression', 'LinearRegression'] ):
        from mlwrap.explainers.shap import LinearSHAP

        return LinearSHAP(
            config=config, model=model, column_transformer=column_transformer, background_data=background_data
        )        

    if 'DecisionTree' in model_type:
        return SklearnDecisionTreeExplainer(
            config=config, model=model, column_transformer=column_transformer, background_data=background_data
        )

    raise NotImplementedError
