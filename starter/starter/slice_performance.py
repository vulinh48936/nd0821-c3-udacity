import sys
import os
import pickle
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference

def get_sliced_performance_metrics(model, encoder, lb, data, slice_feature, categorical_features=[]):
    """
    Calculate and print performance metrics on slices of data based on a specified feature.

    Parameters
    ----------
    model : Machine learning model
        A trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        A trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        A trained sklearn LabelBinarizer, only used if training=False.
    data : pd.DataFrame
        A pandas DataFrame containing the features and label.
    slice_feature : str
        The name of the feature used to make slices (categorical features).
    categorical_features : list[str], optional
        A list containing the names of the categorical features (default=[]).

    Returns
    -------
    None
    """
    # Save the original stdout for later
    original_stdout = sys.stdout

    # Open a file to redirect stdout to
    with open(os.path.join(os.path.dirname(__file__), "slice_output.txt"), "w") as f:
        # Redirect stdout to the file
        sys.stdout = f

        # Print header information
        print(f"Performance on slices of data based on {slice_feature}")
        print("*****************************************************")

        # Process the data
        X, y, _, _ = process_data(data, categorical_features=categorical_features, label="salary", training=True)

        # Make predictions
        preds = inference(model, X)

        # Loop over unique values of the slice feature
        for slice_value in data[slice_feature].unique():
            # Get the index of rows with the current slice value
            slice_index = data.index[data[slice_feature] == slice_value]

            # Print slice information and model metrics
            print(f"{slice_feature} = {slice_value}")
            print(f"data size: {len(slice_index)}")
            precision, recall, fbeta = compute_model_metrics(y[slice_index], preds[slice_index])
            print(f"precision: {precision}, recall: {recall}, fbeta: {fbeta}")
            print("-------------------------------------------------")

        # Reset stdout to the original value
        sys.stdout = original_stdout


if __name__ == '__main__':
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
    file_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(file_dir, '../data/clean_census.csv'))

    model_path = os.path.join(file_dir, '../model/lr_model.pkl')
    model = pickle.load(open(model_path, 'rb'))

    encoder_path = os.path.join(file_dir, '../model/encoder.pkl')
    encoder = pickle.load(open(encoder_path, 'rb'))

    lb_path = os.path.join(file_dir, '../model/lb.pkl')
    lb = pickle.load(open(lb_path, 'rb'))

    get_sliced_performance_metrics(model, encoder, lb, data, 'workclass', categorical_features=cat_features)
