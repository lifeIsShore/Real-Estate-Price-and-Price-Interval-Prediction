# that contain the all price pred min max pred (3 ML model)
# needs a back end becaseu can not receive the followings: population etc (needs a db )

import joblib
import pandas as pd

# Load the model
rf_model_loaded = joblib.load('random_forest_model_compressed.pkl')

# Example: new data for prediction
new_data = pd.DataFrame({
    'number_of_rooms': [4],
    'living_area': [150],
    'land-size': [300],
    'city_score_normalized': [0.7],
    'population': [500000],
    'geo_spatial': [0.5],
    'center_distance': [20],
    'airport_distance': [150]
})

# Predict the price using the loaded model
predicted_price = rf_model_loaded.predict(new_data)

# Print the predicted price
print("Predicted Price:", predicted_price)
