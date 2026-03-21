import pandas as pd
import folium
import webbrowser
import os
from paths import DATA_DIR, ROUTES_DIR

def decode_polyline(encoded):
    points = []
    index = 0
    lat = 0
    lng = 0
    while index < len(encoded):
        b, shift, result = 0, 0, 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        lat += ~(result >> 1) if result & 1 else result >> 1
        shift, result = 0, 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        lng += ~(result >> 1) if result & 1 else result >> 1
        points.append((lat / 1e5, lng / 1e5))
    return points

def build_heatmap():
    csv_path = os.path.join(DATA_DIR, "my_rides.csv")
    if not os.path.exists(csv_path):
        print("No my_rides.csv found.")
        return
    df = pd.read_csv(csv_path)
    df = df[df["polyline"].notna()]

    print(f"Building map from {len(df)} rides...")

    # Center on Iowa City
    m = folium.Map(location=[41.6611, -91.5302], zoom_start=12)

    # Plot each ride as a line
    for _, row in df.iterrows():
        try:
            points = decode_polyline(row["polyline"])
            folium.PolyLine(
                points,
                color="#fc4c02",
                weight=2,
                opacity=0.4,
                tooltip=f"{row['name']} — {row['distance_mi']}mi"
            ).add_to(m)
        except Exception:
            continue

    # Add start marker
    folium.Marker(
        [41.6611, -91.5302],
        popup="Iowa City",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    os.makedirs(ROUTES_DIR, exist_ok=True)
    map_file = os.path.join(ROUTES_DIR, "my_rides_map.html")
    m.save(map_file)
    print(f"Saved to {map_file}")
    webbrowser.open(f"file:///{os.path.abspath(map_file)}")

if __name__ == "__main__":
    build_heatmap()