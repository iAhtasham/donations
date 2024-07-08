import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

class DonationPredictor:
    def __init__(self, data_path='updated.csv'):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.features = ['Zakat', 'Fitrana', 'Sadqa', 'Fidya', 'Flood_Relief', 'Ramazan_Package', 'Kaffara', 'Earthquake',
                         'Khairat', 'Orphan_Care', 'Khummas', 'Education', 'Marriage', 'Old_Age_Home', 'Aqiqah']
        self.ratings = ['Zakat_Rating', 'Fitrana_Rating', 'Sadqa_Rating', 'Fidya_Rating', 'Flood_Relief_Rating',
                        'Ramazan_Package_Rating', 'Kaffara_Rating', 'Earthquake_Rating', 'Khairat_Rating',
                        'Orphan_Care_Rating', 'Khummas_Rating', 'Education_Rating', 'Marriage_Rating',
                        'Old_Age_Home_Rating', 'Aqiqah_Rating']
        self.target = 'Engagement'

    def preprocess_data(self):
        for feature, rating in zip(self.features, self.ratings):
            self.df[feature] = self.df[feature] * self.df[rating]

    def train_model(self, test_size=0.2, random_state=42):
        X = self.df[self.features]
        y = self.df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.rf = RandomForestClassifier(random_state=random_state)
        self.rf.fit(X_train_scaled, y_train)

        predictions = self.rf.predict(X_test_scaled)
        self.accuracy = accuracy_score(y_test, predictions)

    def recommend_campaign(self, row):
        importances = self.rf.feature_importances_
        scores = {feature: row[feature] * row[feature + '_Rating'] * importance
                  for feature, importance in zip(self.features, importances)}
        top_feature = max(scores, key=scores.get)
        return top_feature

    def predict_for_row(self, row):
        row_df = pd.DataFrame([row], columns=self.features + [f + '_Rating' for f in self.features])
        scaled_row = StandardScaler().fit_transform(row_df[self.features])
        return self.rf.predict(scaled_row)

# Example usage:
predictor = DonationPredictor()
predictor.preprocess_data()
predictor.train_model()

# Example of predicting for a single row:
single_row = {'Zakat': 100, 'Fitrana': 50, 'Sadqa': 30, 'Fidya': 10, 'Flood_Relief': 20, 'Ramazan_Package': 40,
              'Kaffara': 60, 'Earthquake': 10, 'Khairat': 30, 'Orphan_Care': 50, 'Khummas': 20, 'Education': 30,
              'Marriage': 40, 'Old_Age_Home': 10, 'Aqiqah': 20,
              'Zakat_Rating': 5, 'Fitrana_Rating': 4, 'Sadqa_Rating': 3, 'Fidya_Rating': 2, 'Flood_Relief_Rating': 4,
              'Ramazan_Package_Rating': 5, 'Kaffara_Rating': 3, 'Earthquake_Rating': 2, 'Khairat_Rating': 4,
              'Orphan_Care_Rating': 5, 'Khummas_Rating': 3, 'Education_Rating': 4, 'Marriage_Rating': 5,
              'Old_Age_Home_Rating': 2, 'Aqiqah_Rating': 3}

prediction = predictor.predict_for_row(single_row)
print("Predicted Engagement:", prediction[0])
