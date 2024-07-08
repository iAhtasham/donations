from flask import Flask, request, jsonify
import pandas as pd
from save import DonationPredictor  # Importing the DonationPredictor class

app = Flask(__name__)

# Initialize the predictor
predictor = DonationPredictor()

# Preprocess data and train the model
predictor.preprocess_data()
predictor.train_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure that all required features and their ratings are provided
        required_features = ['Zakat', 'Fitrana', 'Sadqa', 'Fidya', 'Flood_Relief', 'Ramazan_Package', 'Kaffara',
                             'Earthquake', 'Khairat', 'Orphan_Care', 'Khummas', 'Education', 'Marriage',
                             'Old_Age_Home', 'Aqiqah']
        required_ratings = ['Zakat_Rating', 'Fitrana_Rating', 'Sadqa_Rating', 'Fidya_Rating', 'Flood_Relief_Rating',
                            'Ramazan_Package_Rating', 'Kaffara_Rating', 'Earthquake_Rating', 'Khairat_Rating',
                            'Orphan_Care_Rating', 'Khummas_Rating', 'Education_Rating', 'Marriage_Rating',
                            'Old_Age_Home_Rating', 'Aqiqah_Rating']
        for feature in required_features:
            if feature not in data or f'{feature}_Rating' not in data:
                return jsonify({'error': f'Missing feature or rating for {feature}'}), 400

        # Convert int64 values to regular Python integers
        data = {key: int(value.values[0]) if isinstance(value, pd.Series) else value for key, value in data.items()}

        # Predict engagement
        prediction = predictor.predict_for_row(data)
        print("Prediction: ", prediction[0].item(), "type: ", type(prediction[0].item()))
        return jsonify({'prediction': prediction[0].item()}), 200

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
