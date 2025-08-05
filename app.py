from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yoursecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expenses.db'
db = SQLAlchemy(app)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20))
    amount = db.Column(db.Float)
    category = db.Column(db.String(50))
    desc = db.Column(db.String(400))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    expenses = Expense.query.order_by(Expense.date.desc()).all()
    categories = sorted({e.category for e in expenses})
    return render_template('index.html', expenses=expenses, categories=categories)

@app.route('/add', methods=['POST'])
def add():
    date = request.form['date']
    amount = float(request.form['amount'])
    category = request.form['category']
    desc = request.form['desc']
    e = Expense(date=date, amount=amount, category=category, desc=desc)
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
    df = pd.DataFrame([(d.date, d.amount, d.category, d.desc) for d in data],
                      columns=['date', 'amount', 'category', 'desc'])
    df.to_csv('expenses_export.csv', index=False)
    return send_file('expenses_export.csv', as_attachment=True)

@app.route('/predict')
def predict():
    try:
        model = joblib.load('model.pkl')
        data = Expense.query.all()
        df = pd.DataFrame([(d.date, d.amount, d.category) for d in data],
                          columns=['date', 'amount', 'category'])
        prediction = model.predict([[len(df), df['amount'].sum()]])
        return f"Predicted next month's spend: ₹{prediction[0]:.2f}"
    except Exception as ex:
        return f"ML feature coming soon! ({ex})"

if __name__ == '__main__':
    app.run(debug=True)
