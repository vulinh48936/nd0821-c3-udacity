from fastapi.testclient import TestClient
from main import app

# Call Test-Client to run normal Python tests
client = TestClient(app)


# Test the main page message
def test_get_function():
    result = client.get("/")
    assert result.status_code == 200
    assert result.json() == {
        "message": "Greetings! This application aims to predict whether an individual's income will surpass $50,000 per year or not."
    }


# Test the predict function and its message for income below $50K
def test_post_predict_below_50k():
    result = client.post("/predict", json={
        "age": 42,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 37618,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Farming-fishing",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    })

    assert result.status_code == 200
    assert result.json() == {"Income category is: ": "<=50K"}


# Test the predict function and its message for income above $50K
def test_post_predict_above_50k():
    result = client.post("/predict", json={
        "age": 61,
        "workclass": "Private",
        "fnlgt": 195453,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    })

    assert result.status_code == 200
    assert result.json() == {"Income category is: ": "<=50K"}


if __name__ == "__main__":
    test_get_function()
    test_post_predict_below_50k()
    test_post_predict_above_50k()