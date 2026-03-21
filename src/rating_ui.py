from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from paths import DATA_DIR, OUTPUT_DIR

app = Flask(__name__)

RIDES_FILE      = os.path.join(DATA_DIR, "my_rides.csv")
RATINGS_FILE    = os.path.join(DATA_DIR, "my_ratings.csv")
GENERATED_FILE  = os.path.join(DATA_DIR, "generated_routes.csv")
FEEDBACK_FILE   = os.path.join(DATA_DIR, "feedback_ratings.csv")

def load_rides():
    df = pd.read_csv(RIDES_FILE)
    if os.path.exists(RATINGS_FILE):
        ratings = pd.read_csv(RATINGS_FILE)[["id", "rating"]]
        df = df.merge(ratings, on="id", how="left")
    else:
        df["rating"] = None
    return df

def save_rating(ride_id, rating):
    if os.path.exists(RATINGS_FILE):
        ratings = pd.read_csv(RATINGS_FILE)
        if ride_id in ratings["id"].values:
            ratings.loc[ratings["id"] == ride_id, "rating"] = rating
        else:
            ratings = pd.concat([ratings, pd.DataFrame([{"id": ride_id, "rating": rating}])], ignore_index=True)
    else:
        ratings = pd.DataFrame([{"id": ride_id, "rating": rating}])
    ratings.to_csv(RATINGS_FILE, index=False)

@app.route("/")
def index():
    df = load_rides()
    rated_count = df["rating"].notna().sum() if "rating" in df.columns else 0
    return render_template("rating.html", total=len(df), rated_count=int(rated_count))

@app.route("/api/rides")
def api_rides():
    df = load_rides()
    rides = df.to_dict(orient="records")
    for r in rides:
        for k, v in r.items():
            if isinstance(v, float) and pd.isna(v):
                r[k] = None
    return jsonify(rides)

@app.route("/rate", methods=["POST"])
def rate():
    data = request.get_json()
    save_rating(data["id"], data["rating"])
    df = load_rides()
    rated_count = df["rating"].notna().sum()
    return jsonify({"status": "ok", "rated_count": int(rated_count)})

def load_generated_routes():
    if not os.path.exists(GENERATED_FILE):
        return pd.DataFrame()
    df = pd.read_csv(GENERATED_FILE)
    if os.path.exists(FEEDBACK_FILE):
        ratings = pd.read_csv(FEEDBACK_FILE)[["id", "rating"]]
        df = df.merge(ratings, on="id", how="left")
    else:
        df["rating"] = None
    return df

def save_feedback(route_id, rating):
    if os.path.exists(FEEDBACK_FILE):
        ratings = pd.read_csv(FEEDBACK_FILE)
        if route_id in ratings["id"].values:
            ratings.loc[ratings["id"] == route_id, "rating"] = rating
        else:
            ratings = pd.concat([ratings, pd.DataFrame([{"id": route_id, "rating": rating}])], ignore_index=True)
    else:
        ratings = pd.DataFrame([{"id": route_id, "rating": rating}])
    ratings.to_csv(FEEDBACK_FILE, index=False)

@app.route("/feedback")
def feedback():
    df = load_generated_routes()
    rated_count = df["rating"].notna().sum() if "rating" in df.columns and len(df) > 0 else 0
    return render_template("feedback.html", total=len(df), rated_count=int(rated_count))

@app.route("/api/generated_routes")
def api_generated_routes():
    df = load_generated_routes()
    if df.empty:
        return jsonify([])
    routes = df.to_dict(orient="records")
    for r in routes:
        for k, v in r.items():
            if isinstance(v, float) and pd.isna(v):
                r[k] = None
    return jsonify(routes)

@app.route("/feedback_rate", methods=["POST"])
def feedback_rate():
    data = request.get_json()
    save_feedback(data["id"], data["rating"])
    df = load_generated_routes()
    rated_count = df["rating"].notna().sum() if "rating" in df.columns else 0
    return jsonify({"status": "ok", "rated_count": int(rated_count)})

if __name__ == "__main__":
    print("Opening at http://localhost:5000")
    app.run(debug=True, port=5000)
    