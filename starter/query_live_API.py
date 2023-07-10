import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# data_sample
data_sample = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 141297,
    "education": 'Bachelors',
    "education_num": 9,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": 'United-States'
}

app_url = "https://nd0821-c3-udacity.herokuapp.com/predict"

result = requests.post(app_url, json=data_sample)
assert result.status_code == 200

logging.info("Testing Heroku app using 'requests' library")
logging.info(f"Status code (success=200): {result.status_code}")
logging.info(f"Response body (success= '>50K'): {result.json()}")
