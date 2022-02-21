"""Feature engineers the banking dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_extraction.text import _VectorizerMixin

from collections import Counter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "age",
    "job", 
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]
label_column = "y"

feature_columns_dtype = {
    "age": np.float64,
    "job": str, 
    "marital": str,
    "education": str,
    "default": str,
    "housing": str,
    "loan": str,
    "contact": str,
    "month": str,
    "day_of_week": str,
    "duration": np.float64,
    "campaign": np.float64,
    "pdays": np.float64,
    "previous": np.float64,
    "poutcome": str,
    "emp.var.rate": np.float64,
    "cons.price.idx": np.float64,
    "cons.conf.idx": np.float64,
    "euribor3m": np.float64,
    "nr.employed": np.float64,
}
label_column_dtype = {"y": str}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    
    logger.info("The key contains", print(key))

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/banking-additional-full.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        header=0,
#         names=feature_columns_names + [label_column],
        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    os.unlink(fn)
    
    logger.info("Downloaded df contains %s rows and %s columns", df.shape[0], df.shape[1])
    logger.debug("Defining transformers.")
    numeric_features = ["age", "duration", "campaign", "pdays","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_features = ["job", "marital","education","housing","loan","contact","month","day_of_week","poutcome"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    logger.info("Applying transforms.")
    df['y'] = df['y'].map({'yes':1, 'no':0})
    y = df.pop("y")
    X_pre = preprocess.fit_transform(df)20.
    
    # estimate scale_pos_weight value
#     counter = Counter(y)
#     imbalance_ratio = counter[0] / counter[1]
    
    ### Create functions to retrieve the column names from the "preprocess" transformer
    def get_feature_out(estimator, feature_in):
        if hasattr(estimator,'get_feature_names'):
            if isinstance(estimator, _VectorizerMixin):
                # handling all vectorizers
                return [f'vec_{f}' \
                    for f in estimator.get_feature_names()]
            else:
                return estimator.get_feature_names(feature_in)
        elif isinstance(estimator, SelectorMixin):
            return np.array(feature_in)[estimator.get_support()]
        else:
            return feature_in


    def get_ct_feature_names(ct):
        # handles all estimators, pipelines inside ColumnTransfomer
        # doesn't work when remainder =='passthrough'
        # which requires the input column names.
        output_features = []

        for name, estimator, features in ct.transformers_:
            if name!='remainder':
                if isinstance(estimator, Pipeline):
                    current_features = features
                    for step in estimator:
                        current_features = get_feature_out(step, current_features)
                    features_out = current_features
                else:
                    features_out = get_feature_out(estimator, features)
                output_features.extend(features_out)
            elif estimator=='passthrough':
                output_features.extend(ct._feature_names_in[features])

        return output_features

    X = pd.DataFrame(X_pre, 
                 columns=get_ct_feature_names(preprocess))

    X['y'] = y
#     X['y'] = X['y'].astype(np.float64)
    
    # Move our target column from the first to the last position (column) in the data frame
    temp_cols = list(X.columns)
    temp_cols = [temp_cols[-1]] + temp_cols[:-1]
    X = X[temp_cols]

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
#     np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
