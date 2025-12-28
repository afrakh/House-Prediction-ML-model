from flask import Flask, request, jsonify, render_template
import json
import pickle
import numpy as np

app = Flask(__name__)

# Load model & columns
with open('../models/columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

with open('../models/bangalore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

locations = data_columns[3:]

def get_location_index(location):
    try:
        return data_columns.index(location)
    except:
        return -1

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/get_locations')
def get_locations():
    return jsonify({'locations': locations})

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    data = request.get_json()

    total_sqft = float(data['total_sqft'])
    bhk = int(data['bhk'])
    bath = int(data['bath'])
    location = data['location']

    # Prepare input array
    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk

    loc_index = get_location_index(location)
    if loc_index >= 0:
        x[loc_index] = 1

    # Predict
    price = model.predict([x])[0]
    price = max(0, round(price, 2))  # Clamp negative predictions

    return jsonify({'estimated_price': price})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

