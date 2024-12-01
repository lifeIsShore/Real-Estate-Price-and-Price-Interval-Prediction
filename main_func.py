import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
from airport_dist_score import calculate_airport_distance_log
from far_from_center_score import calculate_distance_address_to_city_center
from geo_score import score_address

# Load the pre-trained model
rf_model_loaded = joblib.load('random_forest_model_compressed.pkl')

# Function to predict the price based on user input
def predict_price():
    address = address_entry.get()
    if not address:
        messagebox.showerror("Error", "Please enter an address.")
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

    # Example of additional data from the user or pre-defined values
    new_data = pd.DataFrame({
        'number_of_rooms': [int(number_of_rooms_entry.get())],
        'living_area': [int(living_area_entry.get())],
        'land-size': [int(land_size_entry.get())],
        'city_score_normalized': [center_score],  # Assuming center score is a normalized feature
        'population': [int(population_entry.get())],
        'geo_spatial': [geo_score],
        'center_distance': [center_score],  # Can be adjusted based on actual distance
        'airport_distance': [airport_score]  # Can be adjusted based on actual distance
    })

    predicted_price = rf_model_loaded.predict(new_data)
    messagebox.showinfo("Predicted Price", f"The predicted property price is: €{predicted_price[0]:,.2f}")

# Create the main window
root = tk.Tk()
root.title("Real Estate Price Prediction")

# Create and place widgets (labels, entries, buttons)
tk.Label(root, text="Enter Property Address:").grid(row=0, column=0, sticky="e")
address_entry = tk.Entry(root, width=50)
address_entry.grid(row=0, column=1)

tk.Label(root, text="Number of Rooms:").grid(row=1, column=0, sticky="e")
number_of_rooms_entry = tk.Entry(root, width=50)
number_of_rooms_entry.grid(row=1, column=1)

tk.Label(root, text="Living Area (m²):").grid(row=2, column=0, sticky="e")
living_area_entry = tk.Entry(root, width=50)
living_area_entry.grid(row=2, column=1)

tk.Label(root, text="Land Size (m²):").grid(row=3, column=0, sticky="e")
land_size_entry = tk.Entry(root, width=50)
land_size_entry.grid(row=3, column=1)

tk.Label(root, text="Population of the Area:").grid(row=4, column=0, sticky="e")
population_entry = tk.Entry(root, width=50)
population_entry.grid(row=4, column=1)

# Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=5, column=0, columnspan=2)

# Run the GUI loop
root.mainloop()
