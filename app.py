from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Load all models from the 'models/' directory
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
    current_expense = data.get("current", None)

    if not category or current_expense is None:
        return jsonify({"error": "Missing 'category' or 'current' expense in request."}), 400

    if category not in models:
        return jsonify({"error": f"No model found for category '{category}'"}), 400

    try:
        current_expense = float(current_expense)
    except ValueError:
        return jsonify({"error": "Invalid expense value. Must be a number."}), 400

    model = models[category]
    future_months = np.arange(0, 6).reshape(-1, 1)
    base_preds = model.predict(future_months)

    # Adjust predictions to start from the user-entered current value
    adjusted_preds = (base_preds - base_preds[0]) + current_expense

    return jsonify({
        "category": category,
        "predictions": adjusted_preds.round(2).tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
