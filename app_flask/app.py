from flask import Flask, request, jsonify
import joblib
import numpy as np
import io
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests


class_labels = {0: 'cloudy', 1: 'rain', 2: 'shine', 3: 'sunrise'}

model_aqi = joblib.load('aqi_classifier.pkl')
model_aqi_prediction = load_model('predict_7days_model.h5')
model_weather = load_model('weather_image_classifier.h5')

scaler_aqi = joblib.load('scaler_forecast_aqi.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "AQI Classification API is running!"

@app.route('/api/aqi/classify', methods=['POST'])
def classify():
    data = request.json

    required_features = ['so2', 'no2', 'pm10', 'pm2_5', 'o3', 'co']
    missing_features = [feature for feature in required_features if feature not in data]
    if missing_features:
        return jsonify({'error': f'Missing features: {missing_features}'}), 400

    input_features = np.array([data[feature] for feature in required_features]).reshape(1, -1)

    prediction = model_aqi.predict(input_features)[0]

    categories = {1: 'Good', 2: 'Fair', 3: 'Moderate', 4: 'Poor', 5: 'Very Poor'}
    predicted_label = categories[prediction]

    return jsonify({'predicted_category': predicted_label})


@app.route('/api/weather/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    img_bytes = file.read()
    img = load_img(io.BytesIO(img_bytes), target_size=(224, 224))

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model_weather.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    probabilities = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}

    combined_rain_prob = probabilities['cloudy'] + probabilities['rain']

    chance_of_rain = min(combined_rain_prob, 1.0) * 100

    return jsonify({
        'predicted_weather': predicted_class,
        'probabilities': probabilities,
        'chance_of_rain': f"{chance_of_rain:.2f}%"
    })

@app.route('/api/weather/predictsevendays', methods=['POST'])
def predict_seven_days():
    data = request.json

    api_key = '858dd300a112a11f1233fdde9e291bb8'
    lat = data['lat']
    lon = data['lon']
    open_weather_url = "http://api.openweathermap.org/data/2.5/air_pollution?lat=" + lat + "&lon=" + lon + "&appid=" + api_key
    
    response = requests.get(open_weather_url)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from OpenWeather'}), 500
    
    data = response.json()['list'][0]['components']


    required_features = ['co', 'so2', 'nh3', 'pm2_5']
    missing_features = [feature for feature in required_features if feature not in data]
    
    if missing_features:
        return jsonify({'error': f'Missing features: {missing_features}'}), 400
    
    input_features = np.array([data[feature] for feature in required_features]).reshape(1, -1)
    input_features = scaler_aqi.transform(input_features)

    predictions = model_aqi_prediction.predict(input_features)
    predictions_list = predictions.flatten().tolist()

    results = []

    for i, prediction in enumerate(predictions_list):
        state = ''
        if prediction <= 1 :
            state = 'Good'
        elif prediction <= 2:
            state = 'Fair'
        elif prediction <= 3:
            state = 'Moderate'
        elif prediction <= 4:
            state = 'Poor'
        else:
            state = 'Very Poor'
        results.append({
            'day': i + 1,
            'predicted_aqi': state,
            'predicted_aqi_value': prediction
        })
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
