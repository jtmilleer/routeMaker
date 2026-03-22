import requests
import json
import pandas as pd
import os
from paths import DATA_DIR

url = "https://mapservices.nps.gov/arcgis/rest/services/cultural_resources/nrhp_locations/MapServer/0/query"
params = {
    "where": "STATE = 'IA'",  # Will filter in memory since sometimes CITY varies
    "outFields": "RESNAME,CITY,STATE",
    "outSR": "4326",
    "f": "json",
    "returnGeometry": "true"
}

all_features = []
offset = 0
limit = 2000
while True:
    params["resultOffset"] = offset
    params["resultRecordCount"] = limit
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)
        offset += limit
    except Exception as e:
        print(f"Error: {e}")
        break

print(f"Total features in IA: {len(all_features)}")
ic_features = []
for f in all_features:
    attrs = f.get("attributes", {})
    city = attrs.get("CITY") or ""
    if "IOWA CITY" in city.upper():
        ic_features.append(f)

print(f"Total features in Iowa City: {len(ic_features)}")

rows = []
for f in ic_features:
    attrs = f.get("attributes", {})
    geom = f.get("geometry", {})
    x = geom.get("x")
    y = geom.get("y")
    if x and y:
        rows.append({
            "name": attrs.get("RESNAME"),
            "lat": y,
            "lng": x
        })

df = pd.DataFrame(rows)
out_path = os.path.join(DATA_DIR, "iowa_city_historic_sites.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} historic sites to {out_path}")
