import logging
from typing import Dict, Type, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
)

from mlwrap import algorithms, preparation
from mlwrap.config import ExplanationResult, MLConfig
from mlwrap.enums import ProblemType


def get_scores(
    problem_type: ProblemType,
    y: Union[np.ndarray, pd.Series],
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[Type[str], float]:
    # note that some metrics are calculated using different averging methods so that we can get aa different view of the data in the resulting single value
    # 'macro' means calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account but rather treats each class equally.
    # 'weighted' means that we weight each class contribution with the number of records. This means that if the data is imbalanced the majority class will dominate the average

    scores = {}
    if problem_type == ProblemType.Classification:
        # check how many classes we have in the actuals and predictions - if either is one then we can't derive recall, precision, f1 and AUC
        actuals_raw = pd.get_dummies(y).to_numpy()
        n_actual_classes = len(np.unique(y))
        n_pred_classes = len(np.unique(y_pred))
        logging.debug(
            f"Class counts: Actuals {n_actual_classes}, Predictions {n_pred_classes}"
        )

        # accuracy: (tp + tn) / (p + n)
        # # NOTE: weighted recall is equivalent to accuracy: (tp + tn) / (p + n)
        scores["recall_weighted"] = accuracy_score(y, y_pred)

        if n_actual_classes > 1:
            # NOTE: macro recall is equivalent to balanced accuracy (the average of recall on classes)
            scores["recall_macro"] = balanced_accuracy_score(y, y_pred)

            # precision tp / (tp + fp)
            # recall: tp / (tp + fn)
            # f1: 2 tp / (2 tp + fp + fn)

            for av in ["macro", "weighted"]:
                precision, _, f1, _ = precision_recall_fscore_support(
                    y, y_pred, average=av
                )
                scores["precision_" + av] = precision
                scores["f1_" + av] = f1

            # ROC AUC
            # return macro and weighted auc values. The snag here is that sklearn averaging ignores binary classes
            # so we need to do the averaging ourselves. Also, the precision_recall_fscore_support function doesn't return
            # supports for individual classes
            supports = precision_recall_fscore_support(y, y_pred)[3]

            roc_auc_macro = 0
            roc_auc_weighted = 0
            pr_auc_macro = 0
            pr_auc_weighted = 0
            # the sklearn roc_auc_score function doesn't appear to work properly for binary problems as it returns the same value for macro or weighted averages
            for n in range(n_actual_classes):
                actuals_raw_class = actuals_raw[:, n]
                n_actuals_raw_class = len(np.unique(actuals_raw_class))
                if n_actuals_raw_class > 1:
                    roc_auc_class = roc_auc_score(actuals_raw_class, y_prob[:, n])
                    pr, re, _ = precision_recall_curve(actuals_raw[:, n], y_prob[:, n])
                    pr_auc_class = auc(re, pr)
                else:
                    roc_auc_class = 0
                    pr_auc_class = 0
                roc_auc_macro += roc_auc_class
                roc_auc_weighted += roc_auc_class * supports[n]
                pr_auc_macro += pr_auc_class
                pr_auc_weighted += pr_auc_class * supports[n]

            roc_auc_macro = roc_auc_macro / n_actual_classes
            roc_auc_weighted = roc_auc_weighted / sum(supports)
            pr_auc_macro = pr_auc_macro / n_actual_classes
            pr_auc_weighted = pr_auc_weighted / sum(supports)
            scores["roc_auc_weighted"] = roc_auc_weighted
            scores["roc_auc_macro"] = roc_auc_macro
            scores["pr_auc_weighted"] = pr_auc_weighted
            scores["pr_auc_macro"] = pr_auc_macro
        else:
            logging.warning(
                f"There must be more than one class in both actuals and predictions"
                " to calculate Precisions, Recall, F1 and AUC: Actuals {n_actual_classes},"
                "Predictions {n_pred_classes}"
            )

    elif problem_type == ProblemType.Regression:
        y_norm = y_pred / y
        y_ones = np.ones(y.shape)
        scores["mean_abs_error"] = mean_absolute_error(y, y_pred)
        scores["median_abs_error"] = median_absolute_error(y, y_pred)
        scores["norm_mean_abs_error"] = mean_absolute_error(y_ones, y_norm)
        scores["norm_median_abs_error"] = median_absolute_error(y_ones, y_norm)
        scores["mean_squared_error"] = mean_squared_error(y, y_pred)
    else:
        raise NotImplementedError

    return scores


def calculate_scores(
    config: MLConfig, model, total_row_count: int, y, y_pred, y_prob=None
) -> Dict[str, float]:
    # evaluate metrics
    # n_total_features = len(model.named_steps["variance_threshold"].feature_names_in_)
    # n_active_features = len(model.named_steps["variance_threshold"].get_feature_names_out())
    #  cleaning isn't working at the moment so let's pass on this
    n_total_features = 0
    n_active_features = n_total_features
    n_inactive_features = n_total_features - n_active_features

    algorithm = model.named_steps["algorithm"]
    iterations = algorithms.get_iterations(algorithm)

    scores = {
        "iterations": int(iterations),
        "total_row_count": total_row_count,
        "active_feature_count": n_active_features,
        "inactive_feature_count": n_inactive_features,
    }

    if config.problem_type == ProblemType.Classification:
        class_ratios = preparation.get_class_ratios(y)
        scores["majority_class_fraction"] = float(list(class_ratios.values())[0])
        scores["minority_class_fraction"] = (float(list(class_ratios.values())[-1]),)

    scores = {
        **scores,
        **get_scores(config.problem_type, y, y_pred, y_prob),
    }
    return scores


def print_scores(scores: Dict[str, float]) -> pd.DataFrame:
    if scores is None:
        return None
    logging.info("Scores:")
    metrics = [s for s in scores]
    values = [[s] for s in scores.values()]
    df = pd.DataFrame(data={"value": values}, index=metrics)
    logging.info(df)
    return df


def print_explanation_result(result: ExplanationResult) -> pd.DataFrame:
    if result is None:
        return None
    logging.info("Feature Importances:")
    colname = ["value"]
    rowname = [fi for fi in result.feature_importances]
    vals = [[value] for value in result.feature_importances.values()]
    df = pd.DataFrame(vals, rowname, colname)
    logging.info(df)
    return df
