import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import src.model


def train_test_model():
    """
    Execute model training
    """
    df = pd.read_csv("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/processed_data/clean_census.csv")
    train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = src.model.process_data(
        train, categorical_features=src.model.get_cat_features(),
        label="salary", training=True
    )
    trained_model = src.model.train_model(X_train, y_train)

    dump(trained_model, "/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/model/model.joblib")
    dump(encoder, "/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/model/encoder.joblib")
    dump(lb, "/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/model/lb.joblib")


