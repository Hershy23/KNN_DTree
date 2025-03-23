from flask import Flask, request, jsonify
import numpy as np
import joblib
from sklearn import datasets

app = Flask(__name__)

dt_model = joblib.load("decision_tree_model.pkl")
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

iris = datasets.load_iris()

@app.route("/")
def home():
    return "Iris Classifier API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        dt_prediction = dt_model.predict(features)[0]
        
        knn_prediction = knn_model.predict(scaler.transform(features))[0]

        result = {
            "Decision Tree Prediction": iris.target_names[dt_prediction],
            "KNN Prediction": iris.target_names[knn_prediction]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
