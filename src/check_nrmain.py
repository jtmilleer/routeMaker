import geopandas as gpd

gdb_path = "c:/Users/jtmil/routeMaker/NRIS_CR_Standards_Public.gdb"
print("Reading NR_Main...")
try:
    main_df = gpd.read_file(gdb_path, layer='NR_Main', ignore_geometry=True)
    states = main_df['State'].dropna().unique()
    matches = [s for s in states if 'IA' == s.upper() or 'IOWA' in s.upper()]
    print("States containing 'IA', 'IOWA':", matches)
    for s in matches:
        ia_df = main_df[main_df['State'] == s]
        print(f"Cities starting with IOWA in {s}:", ia_df[ia_df['City'].str.upper().str.startswith('IOWA', na=False)]['City'].unique())
except Exception as e:
    print(e)
