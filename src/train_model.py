import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from paths import DATA_DIR

RIDES_FILE   = os.path.join(DATA_DIR, "my_rides.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "my_ratings.csv")

FEEDBACK_FILE   = os.path.join(DATA_DIR, "feedback_ratings.csv")
GENERATED_FILE  = os.path.join(DATA_DIR, "generated_routes.csv")

MODEL_FILE   = os.path.join(DATA_DIR, "route_model.pkl")

# ── Load & merge data ─────────────────────────────────────────────────────────

def load_training_data():
    rides   = pd.read_csv(RIDES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    df = rides.merge(ratings, on="id", how="inner")
    df = df[df["rating"].notna()]
    print(f"Training on {len(df)} rated rides")
    return df

# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df):
    features = pd.DataFrame()

    features["distance_mi"]       = df["distance_mi"]
    features["elevation_ft"]      = df["elevation_ft"].fillna(0)
    features["avg_speed_mph"]     = df["avg_speed_mph"].fillna(0)
    features["moving_time_min"]   = df["moving_time_min"].fillna(0)

    # Derived features
    features["elev_per_mile"]     = features["elevation_ft"] / features["distance_mi"].replace(0, 1)
    features["distance_sq"]       = features["distance_mi"] ** 2  # captures sweet spot preference
    features["elevation_sq"]      = features["elevation_ft"] ** 2

    # Optional features — fill with 0 if not present
    features["suffer_score"]      = df.get("suffer_score", pd.Series(0, index=df.index)).fillna(0)
    features["avg_watts"]         = df.get("avg_watts",    pd.Series(0, index=df.index)).fillna(0)
    features["pr_count"]          = df.get("pr_count",     pd.Series(0, index=df.index)).fillna(0)

    return features

# ── Train ─────────────────────────────────────────────────────────────────────

def train(df):
    X = build_features(df)
    y = df["rating"].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Cross-validation score
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_mean_absolute_error")
    mae = -scores.mean()
    print(f"Cross-validated MAE: {mae:.2f} (out of 10)")

    # Feature importance
    print("\nFeature importances:")
    for name, importance in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<25} {importance:.3f}")

    return model, scaler, list(X.columns)

# ── Save ──────────────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_names):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "model":         model,
            "scaler":        scaler,
            "feature_names": feature_names,
        }, f)
    print(f"\nModel saved to {MODEL_FILE}")

# ── Predict (for use in route generator later) ────────────────────────────────

def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def predict_score(route_features: dict) -> float:
    """
    Pass in a dict of route features, get back a predicted rating (1-10).
    Example:
        predict_score({
            "distance_mi": 35,
            "elevation_ft": 1500,
            "avg_speed_mph": 16,
            ...
        })
    """
    bundle = load_model()
    model, scaler, feature_names = bundle["model"], bundle["scaler"], bundle["feature_names"]

    df = pd.DataFrame([route_features])

    # Build same features as training
    features = pd.DataFrame()
    features["distance_mi"]     = df["distance_mi"]
    features["elevation_ft"]    = df.get("elevation_ft", pd.Series([0])).fillna(0)
    features["avg_speed_mph"]   = df.get("avg_speed_mph", pd.Series([0])).fillna(0)
    features["moving_time_min"] = df.get("moving_time_min", pd.Series([0])).fillna(0)
    features["elev_per_mile"]   = features["elevation_ft"] / features["distance_mi"].replace(0, 1)
    features["distance_sq"]     = features["distance_mi"] ** 2
    features["elevation_sq"]    = features["elevation_ft"] ** 2
    features["suffer_score"]    = df.get("suffer_score", pd.Series([0])).fillna(0)
    features["avg_watts"]       = df.get("avg_watts",    pd.Series([0])).fillna(0)
    features["pr_count"]        = df.get("pr_count",     pd.Series([0])).fillna(0)

    # Reorder to match training
    features = features[feature_names]
    X_scaled = scaler.transform(features)
    score = model.predict(X_scaled)[0]

    # Clamp to 1-10
    return round(float(np.clip(score, 1, 10)), 2)


def load_training_data():
    # Original rides + ratings
    rides   = pd.read_csv(RIDES_FILE)
    ratings = pd.read_csv(RATINGS_FILE)
    df = rides.merge(ratings, on="id", how="inner")
    df = df[df["rating"].notna()]
    print(f"Original rated rides: {len(df)}")

    # Add feedback from generated routes
    if os.path.exists(GENERATED_FILE) and os.path.exists(FEEDBACK_FILE):
        generated = pd.read_csv(GENERATED_FILE)
        feedback  = pd.read_csv(FEEDBACK_FILE)
        fb = generated.merge(feedback, on="id", how="inner")
        fb = fb[fb["rating"].notna()]
        if len(fb) > 0:
            print(f"Generated route feedback: {len(fb)}")
            df = pd.concat([df, fb], ignore_index=True)

    print(f"Total training samples: {len(df)}")
    return df

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_training_data()
    model, scaler, feature_names = train(df)
    save_model(model, scaler, feature_names)

    # Quick test — predict score for a sample route
    test = {
        "distance_mi":     35,
        "elevation_ft":    1200,
        "avg_speed_mph":   15,
        "moving_time_min": 140,
        "suffer_score":    0,
        "avg_watts":       0,
        "pr_count":        0,
    }
    print(f"\nSample prediction: {predict_score(test)}/10")