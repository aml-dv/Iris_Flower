from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Convert input to numpy array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Mapping class labels
    species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    predicted_species = species_map[prediction]

    return render_template('output.html', prediction=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
