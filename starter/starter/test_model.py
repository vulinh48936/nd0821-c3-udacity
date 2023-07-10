import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)

from ml.model import train_model, compute_model_metrics, inference

def test_train_model():
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)
    # Check if this is a classification model
    assert isinstance(model, BaseEstimator) and isinstance(model, ClassifierMixin)


def test_compute_model_metrics():
    y, preds = [1, 1, 0], [0, 1, 0]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    # Assert that the metrics are close to the expected value:
    # precision = 1.0, recall = 0.5, fbeta = 0.6667
    assert abs(precision - 1) < 0.01 and abs(recall - 0.5) < 0.01 and abs(fbeta - 0.67) < 0.01


def test_inference():
    X = np.random.rand(10, 5)
    y = np.random.randint(2, size=10)
    model = train_model(X, y)
    preds = inference(model, X)
    # Check if preds.shape is similar to y.shape
    assert y.shape == preds.shape

if __name__ == "__main__":
    test_train_model()
    test_compute_model_metrics()
    test_inference()
