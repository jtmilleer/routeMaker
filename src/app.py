import os
import sys
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory
from paths import ROUTES_DIR, SRC_DIR

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def generate_route():
    data = request.json
    route_type = data.get("type", "regular")
    distance = data.get("distance", 35)
    tolerance = data.get("tolerance", 0.25)
    
    script_map = {
        "regular": "generate_routes.py",
        "hilly": "generate_hilly_routes.py",
        "historic": "generate_historic_routes.py",
        "novel": "generate_novel_routes.py"
    }
    
    script_name = script_map.get(route_type, "generate_routes.py")
    script_path = os.path.join(SRC_DIR, script_name)
    
    cmd = [
        sys.executable, script_path,
        "--distance", str(distance),
        "--tolerance", str(tolerance),
        "--no_browser"
    ]
    
    if route_type in ["hilly", "historic"]:
        hilly_factor = data.get("hilly_factor", 100 if route_type=="hilly" else 0)
        cmd.extend(["--hilly_factor", str(hilly_factor)])
        
    if route_type == "historic":
        downtown_radius = data.get("downtown_radius", 1000)
        cmd.extend(["--downtown_radius", str(downtown_radius)])
        
    if route_type == "novel":
        novelty_factor = data.get("novelty_factor", 5.0)
        cmd.extend(["--novelty_factor", str(novelty_factor)])
        
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return jsonify({"status": "error", "message": f"Generation failed: {e.stderr}"}), 500

    map_html_map = {
        "regular": "top_routes.html",
        "hilly": "top_hilly_routes.html",
        "historic": "top_historic_routes.html",
        "novel": "top_routes.html"
    }
    
    html_file = map_html_map.get(route_type)
    
    return jsonify({
        "status": "success",
        "map_url": f"/maps/{html_file}"
    })

@app.route("/maps/<path:filename>")
def serve_map(filename):
    return send_from_directory(ROUTES_DIR, filename)

if __name__ == "__main__":
    print("Starting Route Maker Web Interface on http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
