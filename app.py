from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    features = {
        'Pclass': request.form['Pclass'],
        'Sex': request.form['Sex'],
        'Age': float(request.form['Age']),
        'SibSp': int(request.form['SibSp']),
        'Parch': int(request.form['Parch']),
        'Fare': float(request.form['Fare']),
        'Embarked': request.form['Embarked'],
    }

    # Convert features to a DataFrame
    input_data = pd.DataFrame([features])

    # One-hot encode the categorical variables
    input_data = pd.get_dummies(input_data, columns=['Sex', 'Embarked'], drop_first=True)

    # Reindex to ensure all columns are present
    input_data = input_data.reindex(columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'], fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
