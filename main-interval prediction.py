# Updated Real Estate Price Prediction Script with Area Adjustment

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
from airport_dist_score import calculate_airport_distance_log
from far_from_center_score import calculate_distance_address_to_city_center
from geo_score import score_address

# Load the pre-trained models
model_folder = "C:\\Users\\ahmty\\Desktop\\HFU\\5 Fünftesemester\\signal processing\\myProject papers\\project\\Real-Estate-Price-and-Price-Interval-Prediction"
rf_model_loaded = joblib.load(f"{model_folder}\\random_forest_model_compressed.pkl")
xgb_max_model_loaded = joblib.load(f"{model_folder}\\xgb_model_compressed_max_model.pkl")
xgb_min_model_loaded = joblib.load(f"{model_folder}\\xgb_model_compressed_min_model.pkl")

# Load the city score and population CSV
data_file = f"{model_folder}\\csvs\\zip-city_score.xlsx"
city_data = pd.read_excel(data_file)

# Function to get city_score and population by zip code
def get_city_data(zip_code):
    try:
        # Filter by ZipCode
        row = city_data.loc[city_data['ZipCode'] == zip_code]
        if not row.empty:
            # Extract city_score and population
            city_score = row['city_score'].values[0]
            population = row['population'].values[0]
            return city_score, population
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching city data: {e}")
        return None, None

# Function to predict the price and interval based on user input
def predict_price():
    address = address_entry.get()
    zip_code = zip_code_entry.get()
    
    if not address or not zip_code:
        messagebox.showerror("Error", "Please enter both an address and a zip code.")
        return

    # Get city_score and population from the CSV
    try:
        zip_code = int(zip_code)  # Ensure the zip code is an integer
        city_score, population = get_city_data(zip_code)
        if city_score is None or population is None:
            messagebox.showerror("Error", "Data not found for the provided zip code.")
            return
    except ValueError:
        messagebox.showerror("Error", "Invalid zip code format. Please enter a numeric zip code.")
        return

    geo_score = score_address(address)
    if geo_score is None:
        messagebox.showerror("Error", "Could not calculate geo score. Check the address.")
        return

    center_score = calculate_distance_address_to_city_center(address)
    if center_score is None:
        messagebox.showerror("Error", "Could not calculate center score. Check the address.")
        return

    airport_score = calculate_airport_distance_log(address)
    if airport_score is None:
        messagebox.showerror("Error", "Could not calculate airport score. Check the address.")
        return

    try:
        living_area = float(living_area_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid numeric value for Living Area.")
        return

    # Prepare data for prediction
    new_data = pd.DataFrame({
        'number_of_rooms': [int(number_of_rooms_entry.get())],
        'living_area': [living_area],
        'land-size': [int(land_size_entry.get())],
        'city_score_normalized': [city_score],  # Use fetched city_score
        'population': [population],  # Use fetched population
        'geo_spatial': [geo_score],
        'center_distance': [center_score],  # Can be adjusted based on actual distance
        'airport_distance': [airport_score]  # Can be adjusted based on actual distance
    })

    # Predict values (multiply by living_area for total price)
    predicted_price_per_m2 = rf_model_loaded.predict(new_data)[0]
    predicted_price = predicted_price_per_m2

    predicted_min_price_per_m2 = xgb_min_model_loaded.predict(new_data)[0]
    predicted_min_price = predicted_min_price_per_m2 * living_area

    predicted_max_price_per_m2 = xgb_max_model_loaded.predict(new_data)[0]
    predicted_max_price = predicted_max_price_per_m2 * living_area

    # Display results
    messagebox.showinfo(
        "Predicted Price Interval",
        f"The predicted property price is approximately: €{predicted_price:,.2f}\n\n"
        f"Price Interval:\n"
        f"Minimum: €{predicted_min_price:,.2f}\n"
        f"Maximum: €{predicted_max_price:,.2f}"
    )

# Create the main window
root = tk.Tk()
root.title("Real Estate Price Prediction")

# Create and place widgets (labels, entries, buttons)
tk.Label(root, text="Enter Property Address:").grid(row=0, column=0, sticky="e")
address_entry = tk.Entry(root, width=50)
address_entry.grid(row=0, column=1)

tk.Label(root, text="Enter Zip Code:").grid(row=1, column=0, sticky="e")
zip_code_entry = tk.Entry(root, width=50)
zip_code_entry.grid(row=1, column=1)

tk.Label(root, text="Number of Rooms:").grid(row=2, column=0, sticky="e")
number_of_rooms_entry = tk.Entry(root, width=50)
number_of_rooms_entry.grid(row=2, column=1)

tk.Label(root, text="Living Area (m²):").grid(row=3, column=0, sticky="e")
living_area_entry = tk.Entry(root, width=50)
living_area_entry.grid(row=3, column=1)

tk.Label(root, text="Land Size (m²):").grid(row=4, column=0, sticky="e")
land_size_entry = tk.Entry(root, width=50)
land_size_entry.grid(row=4, column=1)

# Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=5, column=0, columnspan=2)

# Run the GUI loop
root.mainloop()
