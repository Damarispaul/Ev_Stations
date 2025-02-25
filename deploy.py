from fastapi import FastAPI
import joblib
import pandas as pd

# Load the model correctly using joblib
kmeans_model = joblib.load("kmeans_ev_model.pkl")

# Load saved data
charging_df = pd.read_csv("nairobi_charging_stations.csv")
pois_df = pd.read_csv("nairobi_pois_data.csv")

# Initialize FastAPI
app = FastAPI()

# API to return charging stations
@app.get("/charging_stations")
def get_charging_stations():
    return charging_df.to_dict(orient="records")

# API to return POIs
@app.get("/pois")
def get_pois():
    return pois_df.to_dict(orient="records")

# API to get suggested new charging locations
@app.get("/suggest_new_stations")
def suggest_new_stations():
    new_station_locations = kmeans_model.cluster_centers_
    return [{"Latitude": lat, "Longitude": lon} for lat, lon in new_station_locations]

# Run the API with `uvicorn script_name:app --reload`

# Run below to run the script
# uvicorn deploy:app --reload

