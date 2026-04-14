"""
app.py  —  Flask Web Application
----------------------------------
Run AFTER main.py has trained the model.

Usage:
    python app.py

Then open:  http://127.0.0.1:5000

Routes:
  GET  /        → upload form (index.html)
  POST /pred    → receives uploaded file, runs PredictPipeline, returns results
"""

import os
import glob
import shutil
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline

# ── Temp folder for uploaded files ────────────────────────────────────────────
TEMP_DIR = os.path.join("static", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Clean up any leftover files from previous session
for f in glob.glob(os.path.join(TEMP_DIR, "*")):
    try:
        os.remove(f)
    except OSError:
        pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


@app.route("/")
def index():
    return render_template("index.html", sub=None)


@app.route("/pred", methods=["POST"])
def pred():
    if "file" not in request.files:
        return render_template("index.html", sub="<p style='color:red'>No file selected.</p>")

    f = request.files["file"]

    if f.filename == "":
        return render_template("index.html", sub="<p style='color:red'>No file selected.</p>")

    allowed = {".csv", ".xlsx", ".xls"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return render_template(
            "index.html",
            sub="<p style='color:red'>Only CSV and Excel files are supported.</p>"
        )

    # Save uploaded file
    save_path = os.path.join(TEMP_DIR, f.filename)
    f.save(save_path)

    # Run prediction pipeline
    result = PredictPipeline(save_path)

    return render_template("index.html", sub=result.sub)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
