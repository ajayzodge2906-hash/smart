from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Allow requests from any origin

# Load models
model_dir = 'models'
models = {}

for filename in os.listdir(model_dir):
    if filename.endswith('_model.pkl'):
        category = filename.replace('_model.pkl', '').capitalize()
        with open(os.path.join(model_dir, filename), 'rb') as f:
            models[category] = pickle.load(f)

@app.route("/")
def home():
    return "✅ Smart Expense Predictor API is live with CORS!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    category = data.get("category", "").capitalize()

    if category not in models:
        return jsonify({"error": f"No model found for category '{category}'"}), 400

    model = models[category]
    future_months = np.arange(0, 6).reshape(-1, 1)
    predictions = model.predict(future_months)

    return jsonify({
        "category": category,
        "predictions": predictions.round(2).tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
