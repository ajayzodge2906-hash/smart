import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import pytesseract
from PIL import Image
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yoursecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expenses.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
db = SQLAlchemy(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20))
    amount = db.Column(db.Float)
    category = db.Column(db.String(50))
    desc = db.Column(db.String(400))
    budget = db.Column(db.Float, default=0)
    goal = db.Column(db.Float, default=0)
    receipt = db.Column(db.String(200))
    ocr = db.Column(db.Text)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    expenses = Expense.query.order_by(Expense.date.desc()).all()
    categories = sorted(list({e.category for e in expenses}))
    return render_template('index.html', expenses=expenses, categories=categories)

@app.route('/add', methods=['POST'])
def add():
    date = request.form['date']
    amount = float(request.form['amount'])
    category = request.form['category']
    desc = request.form['desc']
    budget = float(request.form.get('budget', 0))
    goal = float(request.form.get('goal', 0))
    receipt = None
    ocr_result = ''
    if 'receipt' in request.files:
        img = request.files['receipt']
        if img and img.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
            img.save(filepath)
            receipt = filepath
            ocr_result = pytesseract.image_to_string(Image.open(filepath))
    e = Expense(date=date, amount=amount, category=category, desc=desc, budget=budget, goal=goal, receipt=receipt, ocr=ocr_result)
    db.session.add(e)
    db.session.commit()
    flash('Expense Added!')
    return redirect(url_for('index'))

@app.route('/delete/<int:id>')
def delete(id):
    e = Expense.query.get(id)
    if e:
        db.session.delete(e)
        db.session.commit()
    return redirect(url_for('index'))

@app.route('/export')
def export():
    data = Expense.query.all()
    df = pd.DataFrame([(d.date, d.amount, d.category, d.desc, d.ocr) for d in data],
                      columns=['date','amount','category','desc','ocr'])
    df.to_csv('expenses_export.csv', index=False)
    return send_file('expenses_export.csv', as_attachment=True)

@app.route('/predict')
def predict():
    try:
        model = joblib.load('model.pkl')
        data = Expense.query.all()
        df = pd.DataFrame([(d.date, d.amount, d.category) for d in data],
                          columns=['date','amount','category'])
        prediction = model.predict([[len(df), df['amount'].sum()]])
        return f"Predicted next month's spend: ₹{prediction[0]:.2f}"
    except Exception as ex:
        return f"ML feature coming soon! ({ex})"

@app.route('/api/summary')
def summary():
    # Example JSON API for dashboard
    q = db.session.query(Expense.category, db.func.sum(Expense.amount)).group_by(Expense.category).all()
    return jsonify({c:a for c,a in q})

if __name__ == '__main__':
    app.run(debug=True)
