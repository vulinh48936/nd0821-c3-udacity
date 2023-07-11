import os
import uvicorn
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from starter.ml.model import inference
from starter.ml.data import process_data

# Set up DVC in Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -df s3-bucket s3://myherukobucket")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Create the application
app = FastAPI()

class UserClassInfo(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', 'Self-emp-inc', 'Without-pay'
    ]
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate',
        'Assoc-voc', 'Prof-school', '5th-6th', '10th', 'Preschool', '12th', '1st-4th'
    ]
    education_num: int
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Separated', 'Married-AF-spouse', 'Widowed'
    ]
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
        'Other-service', 'Sales', 'Transport-moving', 'Farming-fishing',
        'Machine-op-inspct', 'Tech-support', 'Craft-repair', 'Protective-serv',
        'Armed-Forces', 'Priv-house-serv'
    ]
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'
    ]
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ]
    sex: Literal[
        'Male', 'Female'
    ]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico', 'Puerto-Rico',
        'Honduras', 'England', 'Canada', 'Germany', 'Iran', 'Philippines', 'Poland',
        'Columbia', 'Cambodia', 'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti',
        'Portugal', 'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago', 'Greece',
        'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'
    ]

    class Config:
        # This is an example with less than 50k response
        schema_extra = {
            "example": {
                "age": 61,
                "workclass": 'Private',
                "fnlgt": 195453,
                "education": 'HS-grad',
                "education_num": 9,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 50,
                "native_country": 'United-States'
            }
        }


# Load model, encoder, and lb
classifier = joblib.load("./starter/model/lr_model.pkl")
encoder = joblib.load("./starter/model/encoder.pkl")
lb = joblib.load("./starter/model/lb.pkl")


# Define GET function
@app.get("/")
async def get_items():
    return {"message": "Greetings! This application aims to predict whether an individual's income will surpass $50,000 per year or not."}


# Create a POST function that predicts the income category based on the input of all 15 features.
@app.post("/predict")
async def predict(data_input: UserClassInfo):
    # Define categorical features, as they are necessary to process the input data and create accurate features that align with those used during training.
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

    # Create input data with the right format
    test_input_from_web = np.array([[
        data_input.age,
        data_input.workclass,
        data_input.fnlgt,
        data_input.education,
        data_input.education_num,
        data_input.marital_status,
        data_input.occupation,
        data_input.relationship,
        data_input.race,
        data_input.sex,
        data_input.capital_gain,
        data_input.capital_loss,
        data_input.hours_per_week,
        data_input.native_country
    ]])

    # Transform the web input data into a Pandas DataFrame.
    df_input_data = pd.DataFrame(data=test_input_from_web, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country"
    ])
    x_data, _, _, _ = process_data(
        df_input_data,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False
    )

    # Execute inference on the provided data sample.
    y_prediction = inference(classifier, x_data)
    # Convert the prediction to the label
    y = lb.inverse_transform(y_prediction)[0]
    return {"Income category is: ": y}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
