import fiona
import geopandas as gpd
import pandas as pd
import os
from paths import DATA_DIR

gdb_path = "c:/Users/jtmil/routeMaker/NRIS_CR_Standards_Public.gdb"

print("Reading NR_Main...")
# NR_Main is a non-spatial table but geopandas can read it with ignore_geometry=True
main_df = gpd.read_file(gdb_path, layer='NR_Main', ignore_geometry=True)

ic_main = main_df[
    (main_df['City'].fillna('').str.upper().str.contains('IOWA CITY')) &
    (main_df['State'].fillna('').str.upper() == 'IOWA')
]

print(f"Found {len(ic_main)} sites in NR_Main for Iowa City.")

target_refnums = set(ic_main['PropertyID'].astype(str))
name_map = dict(zip(ic_main['PropertyID'].astype(str), ic_main['Resource_Name']))

point_layers = ['crbldg_pt', 'crsite_pt', 'crdist_pt', 'crobj_pt', 'crstru_pt']

all_points = []
for layer in point_layers:
    try:
        print(f"Reading layer {layer}...")
        gdf = gpd.read_file(gdb_path, layer=layer)
        if 'NRIS_Refnum' not in gdf.columns:
            print(f"Skipping {layer}, no NRIS_Refnum column")
            continue
            
        filtered = gdf[gdf['NRIS_Refnum'].astype(str).isin(target_refnums)]
        
        if len(filtered) == 0:
            continue
            
        print(f"  Found {len(filtered)} geometries in {layer}")
        
        filtered = filtered.to_crs(epsg=4326)
        for _, row in filtered.iterrows():
            geom = row.geometry
            if geom and not geom.is_empty:
                refnum = str(row['NRIS_Refnum'])
                all_points.append({
                    "name": name_map.get(refnum, row.get('RESNAME', 'Unknown')),
                    "lat": geom.y,
                    "lng": geom.x,
                    "refnum": refnum,
                    "layer": layer
                })
    except Exception as e:
        print(f"Error on {layer}: {e}")

df_points = pd.DataFrame(all_points)
os.makedirs(DATA_DIR, exist_ok=True)
out_csv = os.path.join(DATA_DIR, "iowa_city_historic_sites.csv")
df_points.to_csv(out_csv, index=False)
print(f"Total points saved: {len(df_points)} to {out_csv}")
