# that contain the all price pred min max pred (3 ML model)
# needs a back end becaseu can not receive the followings: population etc (needs a db )
import pandas as pd
import joblib
from airport_dist_score import calculate_airport_distance_log
from far_from_center_score import calculate_distance_address_to_city_center
from geo_score import score_address

# Load the pre-trained models for min and max price (XGBoost) and Random Forest
xgb_model_min_loaded = joblib.load('xgb_model_compressed_min_model.pkl')
xgb_model_max_loaded = joblib.load('xgb_model_compressed_max_model.pkl')
rf_model_loaded = joblib.load('random_forest_model_compressed.pkl')

# Mock data (Replace these with real data or use values you want to test)
address = "rottweiler str 78056 villingen schwenningen"
number_of_rooms = 3
living_area = 100  # in m²
land_size = 500  # in m²
population = 50000  # Example population of the area

# Calculate geo, center, and airport scores (simulating these functions)
geo_score = score_address(address)
if geo_score is None:
    print("Error: Could not calculate geo score.")
    exit()

center_score = calculate_distance_address_to_city_center(address)
if center_score is None:
    print("Error: Could not calculate center score.")
    exit()

airport_score = calculate_airport_distance_log(address)
if airport_score is None:
    print("Error: Could not calculate airport score.")
    exit()

# Example of additional data from the user or pre-defined values
new_data = pd.DataFrame({
    'number_of_rooms': [number_of_rooms],
    'living_area': [living_area],
    'land-size': [land_size],
    'city_score_normalized': [center_score],  # Assuming center score is a normalized feature
    'population': [population],
    'geo_spatial': [geo_score],
    'center_distance': [center_score],  # Can be adjusted based on actual distance
    'airport_distance': [airport_score]  # Can be adjusted based on actual distance
})

# Print the new_data for verification
print("New Data for Prediction:")
print(new_data)

# Predict the min and max price using the XGBoost models
predicted_min_price_xgb = xgb_model_min_loaded.predict(new_data)[0] * living_area  # Min price prediction (XGBoost)
predicted_max_price_xgb = xgb_model_max_loaded.predict(new_data)[0] * living_area  # Max price prediction (XGBoost)

# Predict the property price using the Random Forest model
predicted_price_rf = rf_model_loaded.predict(new_data)[0]  # Price prediction (Random Forest)

# Print the predicted results
print(f"Price Range using XGBoost: €{predicted_min_price_xgb:,.2f} - €{predicted_max_price_xgb:,.2f}")
print(f"The predicted property price using Random Forest is: €{predicted_price_rf:,.2f}")
