import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import src.model
import logging


def check_score():
    """
    Execute score checking
    """
    df = pd.read_csv("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/processed_data/clean_census.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/model/model.joblib")
    encoder = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/model/encoder.joblib")
    lb = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/model/lb.joblib")

    slice_values = []

    for cat in src.model.get_cat_features():
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = src.model.process_data(
                df_temp,
                categorical_features=src.model.get_cat_features(),
                label="salary", encoder=encoder, lb=lb, training=False)
            # inference
            y_preds = src.model.inference(trained_model, X_test)

            prc, rcl, fb = src.model.compute_model_metrics(y_test,
                                                                      y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
            logging.info(line)
            slice_values.append(line)

    with open('../data/model/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')