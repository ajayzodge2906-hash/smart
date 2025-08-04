from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models from 'models' directory
model_dir = 'models'
models = {}

for filename in os.listdir(model_dir):
    if filename.endswith('_model.pkl'):
        category = filename.replace('_model.pkl', '').capitalize()
        with open(os.path.join(model_dir, filename), 'rb') as f:
            models[category] = pickle.load(f)

@app.route("/")
def home():
    return "✅ Smart Expense Predictor API is live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    category = data.get("category", "").capitalize()
    current_expense = data.get("current", None)

    if not category or current_expense is None:
        return jsonify({"error": "Missing 'category' or 'current' expense."}), 400

    if category not in models:
        return jsonify({"error": f"No model found for category '{category}'."}), 400

    try:
        current_expense = float(current_expense)
    except ValueError:
        return jsonify({"error": "Invalid expense value."}), 400

    model = models[category]
    next_month = np.array([[1]])  # Predicting for 1 month ahead
    prediction = model.predict(next_month)[0]

    # Adjust based on current value if needed
    base_month_0 = model.predict([[0]])[0]
    adjusted_prediction = (prediction - base_month_0) + current_expense

    return jsonify({
        "category": category,
        "next_month_prediction": round(adjusted_prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
