import pandas as pd
import pytest 
import src.cleaning 


@pytest.fixture 
def data():
    """
    let's get the data
    """
    df = pd.read_csv("/Users/rushikeshnaik/Desktop/MLOPs_Projects/dvc_proj_udacity_3/dvc_proj/data/raw_data/census.csv")
    df= src.cleaning.__clean_dataset(df)

    return df


def test_for_null(data):
    "data null free"

    assert data.shape == data.dropna().shape


def test_question_mark(data):

    """
    Data is assumed to have no question marks values
    """

    assert "?" not in data.values


def test_removed_columns(data):
    """checking for removal of columns """

    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns

