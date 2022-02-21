
"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import auc, precision_score, recall_score, roc_auc_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
#     df.drop('y', axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)
    ######

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)
    predictions = np.round(predictions)
    logger.info("Predictions are of data type %s ", predictions.dtype)
    logger.info("Y Test are of data type %s ", y_test.dtype)

    logger.debug("Calculating mean squared error.")
    auc = roc_auc_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    report_dict = {
        "binary_classification_metrics": {
            "auc": {"value": auc,
                   "standard_deviation" : "NaN"},
            "precision": {"value": precision,
                         "standard_deviation" : "NaN"},
            "recall": {"value": recall,
                      "standard_deviation" : "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with precision: %f and Recall: %f and an AUC of %f " , precision, recall, auc)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))