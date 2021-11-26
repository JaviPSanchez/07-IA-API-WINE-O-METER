print('Importing libraries...')
import joblib
from flask import Flask, request, json, jsonify, render_template
from werkzeug.exceptions import HTTPException
print('Done!')
print()
print('Loading model...')
MODEL_PATH = 'models/model.joblib'
print('Done!')
print()
print('Loading server app...')
app = Flask(__name__)
print('Done!')
print()

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors (which is the basic error
    response with Flask).
    """
    # Start with the correct headers and status code from the error
    response = e.get_response()
    # Replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

class MissingKeyError(HTTPException):
    # We can define our own error for the missing key
    code = 422
    name = "Missing key error"
    description = "JSON content missing key 'input'."

class MissingJSON(HTTPException):
    # We can define our own error for missing JSON
    code = 400
    name = "Missing JSON"
    description = "Missing JSON."

def make_prediction(input: float):
    # Load model
    classifier = joblib.load(MODEL_PATH)
    # Make prediction (the model expects a 2D array that is why we put input in a list of list) and return it
    prediction = classifier.predict([[input]])
    return prediction[0]


@app.route("/predict", methods=["GET"])
def predict():
    # Check parameters
    if request.json:
        # Get JSON as dictionnary
        json_input = request.get_json()
        if "input" not in json_input:
            raise MissingKeyError()
        prediction = make_prediction(float(json_input["input"]))
        response = {
            "quality": str(prediction),
        }
        return jsonify(response), 200
    raise MissingJSON()


    

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=4000)