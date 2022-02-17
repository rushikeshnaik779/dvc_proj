import logging
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
import numpy as np
from numpy import mean, std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def get_cat_features():
    """
    Return feature categories 
    """
    cat_features = [
        "workclass","education", "marital-status", 
        "occupation", "relationship", "race", "sex", "native-country"
    ]

    return cat_features


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    
    else: 
        y = np.array([])
    
    X_categorical = X[categorical_features].values
    X_continous = X.drop(*[categorical_features], axis=1)


    if training is True: 
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()

    
    else: 
        X_categorical = encoder.fit_transform(X_categorical)

        try: 
            y = lb.transform(y.values).ravel()

        except AttributeError:
            pass

    X = np.concatenate([X_continous, X_categorical], axis=1)

    return X, y, encoder, lb

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring="accuracy", 
    cv = cv , n_jobs = -1)

    logging.info('Accuracy : %.3f (%.3f)' %(mean(scores), std(scores)))

    return model 


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_preds = model.predict(X)
    return y_preds
