import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Add the necessary imports for the starter code.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


def train_ml_model():
    # Load the csv data
    logging.info("Get the dataset from the location of the data folder in the repo's root")
    data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, random_state=42, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train and save a model.
    logging.info("Training model")
    rf_model = train_model(X_train, y_train)

    # Scoring
    logging.info("Scoring on test set")
    y_pred = inference(rf_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    logging.info(f"Precision: {precision:.2f}. Recall: {recall:.2f}. Fbeta: {fbeta:.2f}")

    # Save artifacts
    model_path = os.path.join(file_dir, '../model/lr_model.pkl')
    pickle.dump(rf_model, open(model_path, 'wb'))

    encoder_path = os.path.join(file_dir, '../model/encoder.pkl')
    pickle.dump(encoder, open(encoder_path, 'wb'))

    lb_path = os.path.join(file_dir, '../model/lb.pkl')
    pickle.dump(lb, open(lb_path, 'wb'))

    # Predict
    preds = inference(rf_model, X_test)
    logging.info('precision: {}, recall: {}, fbeta: {}'.format(
        *compute_model_metrics(y_test, preds)
    ))


if __name__ == "__main__":
    train_ml_model()
