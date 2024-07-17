from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model with a warning filter
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    model = joblib.load("diamond_price_predictor_model.pkl")

# Mapping for color and clarity
color_mapping = {"D": 1, "E": 2, "F": 3, "G": 4, "H": 5, "I": 6, "J": 7}
clarity_mapping = {"IF": 1, "VVS1": 2, "VVS2": 3, "VS1": 4, "VS2": 5, "SI1": 6, "SI2": 7, "I1": 8}

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        carat = float(request.form['carat'])
        color = request.form['color']
        clarity = request.form['clarity']

        color_num = color_mapping.get(color, 0)
        clarity_num = clarity_mapping.get(clarity, 0)

        prediction = model.predict([[carat, color_num, clarity_num]])

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
