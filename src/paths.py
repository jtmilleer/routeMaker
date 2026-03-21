import os

# Root directory of the project (one level up from src)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR   = os.path.join(ROOT_DIR, "data")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
ROUTES_DIR = os.path.join(OUTPUT_DIR, "routes")

# Create essential directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROUTES_DIR, exist_ok=True)
