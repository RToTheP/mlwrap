import logging
from typing import Callable, Dict, List, Type

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

from mlwrap.config import MLConfig, ModelScore
from mlwrap.data.config import DataDetails
from mlwrap.data.encoders import EncoderBase
from mlwrap.enums import ProblemType, ScoreType


def get_scores(
    config: MLConfig,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    actuals: np.ndarray,
    encoders: Dict[str, EncoderBase],
):
    # note that some metrics are calculated using different averging methods so that we can get aa different view of the data in the resulting single value
    # 'macro' means calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account but rather treats each class equally.
    # 'weighted' means that we weight each class contribution with the number of records. This means that if the data is imbalanced the majority class will dominate the average

    scores: List[Type[ModelScore]] = list()
    if config.problem_type == ProblemType.Classification:
        # predict probabilities for test set
        predicted_probabilities: np.ndarray = predict_fn(inputs)
        # predict crisp classes for test set
        predicted_classes = np.argmax(predicted_probabilities, axis=1)
        actuals_raw = actuals
        actuals = np.argmax(actuals, axis=1)

        # check how many classes we have in the actuals and predictions - if either is one then we can't derive recall, precision, f1 and AUC
        n_actual_classes = len(np.unique(actuals))
        n_pred_classes = len(np.unique(predicted_classes))
        logging.debug(
            f"Class counts: Actuals {n_actual_classes}, Predictions {n_pred_classes}"
        )

        # accuracy: (tp + tn) / (p + n)
        # # NOTE: weighted recall is equivalent to accuracy: (tp + tn) / (p + n)
        scores.append(
            ModelScore(
                ScoreType.recall_weighted, accuracy_score(actuals, predicted_classes)
            )
        )

        if n_actual_classes > 1:
            # NOTE: macro recall is equivalent to balanced accuracy (the average of recall on classes)
            scores.append(
                ModelScore(
                    ScoreType.recall_macro,
                    balanced_accuracy_score(actuals, predicted_classes),
                )
            )

            # precision tp / (tp + fp)
            # recall: tp / (tp + fn)
            # f1: 2 tp / (2 tp + fp + fn)

            for av in ["macro", "weighted"]:
                precision, _, f1, _ = precision_recall_fscore_support(
                    actuals, predicted_classes, average=av
                )
                scores.append(ModelScore(ScoreType["precision_" + av], precision))
                scores.append(ModelScore(ScoreType["f1_" + av], f1))

            # ROC AUC
            # return macro and weighted auc values. The snag here is that sklearn averaging ignores binary classes
            # so we need to do the averaging ourselves. Also, the precision_recall_fscore_support function doesn't return
            # supports for individual classes
            supports = precision_recall_fscore_support(actuals, predicted_classes)[3]

            roc_auc_macro = 0
            roc_auc_weighted = 0
            pr_auc_macro = 0
            pr_auc_weighted = 0
            # the sklearn roc_auc_score function doesn't appear to work properly for binary problems as it returns the same value for macro or weighted averages
            for n in range(n_actual_classes):
                actuals_raw_class = actuals_raw[:, n]
                n_actuals_raw_class = len(np.unique(actuals_raw_class))
                if n_actuals_raw_class > 1:
                    roc_auc_class = roc_auc_score(
                        actuals_raw_class, predicted_probabilities[:, n]
                    )
                    pr, re, _ = precision_recall_curve(
                        actuals_raw[:, n], predicted_probabilities[:, n]
                    )
                    pr_auc_class = auc(re, pr)
                else:
                    roc_auc_class = 0
                    pr_auc_class = 0
                roc_auc_macro = roc_auc_macro + roc_auc_class
                roc_auc_weighted = roc_auc_weighted + roc_auc_class * supports[n]
                pr_auc_macro = pr_auc_macro + pr_auc_class
                pr_auc_weighted = pr_auc_weighted + pr_auc_class * supports[n]

            roc_auc_macro = roc_auc_macro / n_actual_classes
            roc_auc_weighted = roc_auc_weighted / sum(supports)
            pr_auc_macro = pr_auc_macro / n_actual_classes
            pr_auc_weighted = pr_auc_weighted / sum(supports)
            scores.append(ModelScore(ScoreType.roc_auc_weighted, roc_auc_weighted))
            scores.append(ModelScore(ScoreType.roc_auc_macro, roc_auc_macro))
            scores.append(ModelScore(ScoreType.pr_auc_weighted, pr_auc_weighted))
            scores.append(ModelScore(ScoreType.pr_auc_macro, pr_auc_macro))
        else:
            logging.warning(
                f"There must be more than one class in both actuals and predictions"
                " to calculate Precisions, Recall, F1 and AUC: Actuals {n_actual_classes},"
                "Predictions {n_pred_classes}"
            )

    elif config.problem_type == ProblemType.Regression:
        pred_values = predict_fn(inputs)

        # decode the values before calculating metrics
        encoder = encoders[config.model_feature_id]
        pred_values = encoder.inverse_transform(pred_values)
        actuals = encoder.inverse_transform(actuals)

        scores.append(
            ModelScore(
                ScoreType.mean_absolute_error, mean_absolute_error(actuals, pred_values)
            )
        )
        scores.append(
            ModelScore(
                ScoreType.median_absolute_error,
                median_absolute_error(actuals, pred_values),
            )
        )
        scores.append(
            ModelScore(
                ScoreType.mean_squared_error, mean_squared_error(actuals, pred_values)
            )
        )
    else:
        raise NotImplementedError

    return scores


def calculate_scores(
    config: MLConfig,
    iterations: int,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    data_details: DataDetails,
) -> List[ModelScore]:
    # evaluate metrics
    n_active_features: int = len([x for x in config.features if x.active])
    scores: List[Type[ModelScore]] = [
        ModelScore(ScoreType.iterations, int(iterations)),
        ModelScore(ScoreType.total_row_count, data_details.total_row_count),
        ModelScore(ScoreType.active_feature_count, n_active_features),
        ModelScore(
            ScoreType.inactive_feature_count, len(config.features) - n_active_features
        ),
    ]

    if config.problem_type == ProblemType.Classification:
        scores.append(
            ModelScore(
                ScoreType.majority_class_fraction,
                float(list(data_details.class_ratios.values())[0]),
            )
        )
        scores.append(
            ModelScore(
                ScoreType.minority_class_fraction,
                float(list(data_details.class_ratios.values())[-1]),
            )
        )

    if data_details.test_input is not None:
        scores.extend(
            get_scores(
                config,
                predict_fn,
                data_details.test_input,
                data_details.test_output,
                data_details.encoders,
            )
        )
    else:
        scores.extend(
            get_scores(
                config,
                predict_fn,
                data_details.train_input,
                data_details.train_output,
                data_details.encoders,
            )
        )
    return scores


def print_scores(scores: List[Type[ModelScore]]):
    if scores is None:
        return
    logging.info("Scores:")
    colname = []
    rowname = []
    vals = []
    first = True
    for m in scores:
        valrow = [m.value]
        if first:
            colname.append("value")
        rowname.append(m.id.name)
        if m.av is not None:
            if first:
                colname.extend(["av", "min", "max", "std", "std_av"])
            valrow.extend([m.av, m.min, m.max, m.std, m.std_av])
        first = False
        vals.append(valrow)
    logging.info(pd.DataFrame(vals, rowname, colname))
