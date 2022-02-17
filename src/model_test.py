from base64 import encode
import pandas as pd 
import numpy as np
import pytest 
from pandas.core.frame import DataFrame
import src.model
from joblib import load


@pytest.fixture 

def data():
    "dataset"

    df = pd.read_csv("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/processed_data/clean_census.csv")

    return df



def test_process_data(data):
    "check split have same number of rows for X & Y"

    encoder = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/encoder.joblib")
    lb = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/lb.joblib")

    X_test, y_test, _, _ = src.model.process_data(
        data, 
        categorical_features=src.model.get_cat_features(),
        label="salary", encoder=encoder, lb=lb, training=False
    )

    assert len(X_test) == len(y_test)



def test_process_encoder(data):
    """
    Check split have same number of rows for X and y
    """


    encoder_test = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/encoder.joblib")
    lb_test = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/lb.joblib")


    _, _, encoder, lb = src.model.process_data(
        data, 
        categorical_features = src.model.get_cat_features(), 
        label="salary", 
        training=True
    )


    _, _, _, _, = src.model.process_data(
        data, 
        categorical_features=src.model.get_cat_features(), 
        label="salary", 
        encoder=encoder_test, lb=lb_test, training=False
    )


    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()




def test_inference_above():
    """
    check inference performance
    """
    model = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/model.joblib")
    encoder = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/encoder.joblib")
    lb = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/lb.joblib")


    array = np.array([[
        32, 
        "Private", 
        "some-college", 
        "Married-civ-spouse", 
        "Exec-managerial", 
        "Husband", 
        "Black", 
        "Male", 
        80, 
        "United-States"
    ]])


    df_temp = DataFrame(data=array, columns=[
        "age", 
        "workclass", 
        "education", 
        "marital-status", 
        "occupation", 
        "relationship", 
        "race", 
        "sex", 
        "hours-per-week", 
        "native-country",
    ])

    X, _, _, _ = src.model.process_data(
        df_temp, 
        categorical_features=src.model.get_cat_features(), 
        encoder = encoder, lb=lb, 
        training=False
    )

    pred = src.model.inference(model, X)

    y = lb.inverse_transform(pred)[0]

    assert y==">50K"



def test_inference_below():
    "Check inference performance"
    encoder_test = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/encoder.joblib")
    lb_test = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/lb.joblib")
    model = load("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/data/model/model.joblib")

    array = np.array([[
        19, 
        "Private", 
        "HS-grad", 
        "Never-married", 
        "Own-child", 
        "Husband", 
        "Black", 
        "Male", 
        40, 
        "United-States"
    ]])


    df_temp = DataFrame(data=array, columns=[
        "age", 
        "workclass", 
        "education", 
        "martial-status", 
        "occupation", 
        "relationship",
        "race", 
        "sex", 
        "hours-per-week",
        "native-country"
    ])


    X, _, _, _ = src.model.process_data(
        df_temp, 
        categorical_features=src.model.get_cat_features(),
        encoder = encoder, lb = lb, training=False
    )
    pred = src.model.inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y=="<50K"