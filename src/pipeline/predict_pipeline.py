"""
Predict Pipeline
----------------
Used by app.py.
Takes a path to an uploaded CSV or Excel file, runs predictions on the
"Claim Description" column, and returns an HTML summary string that
includes:
  - A styled predictions table
  - A bar chart (Coverage Code distribution)
  - A word-cloud image of most common claim words
  - A horizontal bar chart of Accident Source breakdown (if column exists)
"""

import os
import sys
import re
import base64
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                         # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import logger
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import clean_text

ARTIFACTS = "artifacts"
MODEL_PATH      = os.path.join(ARTIFACTS, "best_model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACTS, "tfidf_vectorizer.pkl")
ENCODER_PATH    = os.path.join(ARTIFACTS, "label_encoder.pkl")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _bar_chart(series: pd.Series, title: str, color_palette="Set2") -> str:
    counts = series.value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = sns.color_palette(color_palette, len(counts))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Number of Claims")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(ax.patches, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=9
        )
    fig.tight_layout()
    return _fig_to_b64(fig)


def _horiz_bar(series: pd.Series, title: str) -> str:
    counts = series.value_counts()
    fig, ax = plt.subplots(figsize=(7, max(3, len(counts) * 0.5)))
    colors = sns.color_palette("pastel", len(counts))
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Number of Claims")
    for i, val in enumerate(counts.values[::-1]):
        ax.text(val + 0.1, i, str(val), va="center", fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _wordcloud_chart(texts: pd.Series) -> str:
    """Generate a word-frequency bar chart (no wordcloud library needed)."""
    from collections import Counter
    all_words = " ".join(texts.apply(clean_text))
    word_counts = Counter(all_words.split())
    top = pd.Series(dict(word_counts.most_common(20)))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("Blues_r", len(top))
    ax.barh(top.index[::-1], top.values[::-1], color=colors)
    ax.set_title("Top 20 Most Frequent Words in Claims", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Frequency")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _img_tag(b64: str) -> str:
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;margin:10px 0">'


# ── Main Pipeline ──────────────────────────────────────────────────────────────

class PredictPipeline:
    def __init__(self, file_path: str):
        self.sub = self._run(file_path)

    def _run(self, file_path: str) -> str:
        logger.info(f"Predict pipeline started for: {file_path}")
        try:
            # ── Load artifacts ────────────────────────────────────────────
            model      = load_object(MODEL_PATH)
            vectorizer = load_object(VECTORIZER_PATH)
            le         = load_object(ENCODER_PATH)

            # ── Read uploaded file ────────────────────────────────────────
            if file_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            if "Claim Description" not in df.columns:
                return "<p style='color:red'><b>Error:</b> File must have a <code>Claim Description</code> column.</p>"

            df = df.copy()
            df["Claim Description"] = df["Claim Description"].astype(str).str.strip()
            df = df[df["Claim Description"] != ""]

            # ── Predict ───────────────────────────────────────────────────
            clean  = df["Claim Description"].apply(clean_text)
            X      = vectorizer.transform(clean)
            y_pred = model.predict(X)
            df["Predicted Coverage Code"] = le.inverse_transform(y_pred)

            # ── Build HTML output ─────────────────────────────────────────
            html_parts = []

            # Summary banner
            html_parts.append(f"""
            <div style="background:#e8f5e9;border-left:4px solid #43a047;padding:12px 16px;
                        border-radius:6px;margin-bottom:16px">
              <b>✅ Prediction Complete</b> &nbsp;|&nbsp;
              {len(df)} claims processed &nbsp;|&nbsp;
              {df['Predicted Coverage Code'].nunique()} unique coverage codes found
            </div>""")

            # Coverage Code bar chart
            html_parts.append("<h3 style='margin-top:24px'>📊 Coverage Code Distribution</h3>")
            html_parts.append(_img_tag(_bar_chart(df["Predicted Coverage Code"], "Predicted Coverage Code Distribution")))

            # Accident Source chart (if column present)
            if "Accident Source" in df.columns:
                html_parts.append("<h3 style='margin-top:20px'>⚡ Accident Source Breakdown</h3>")
                html_parts.append(_img_tag(_horiz_bar(df["Accident Source"], "Accident Source Breakdown")))

            # Word frequency chart
            html_parts.append("<h3 style='margin-top:20px'>🔤 Most Common Words in Claims</h3>")
            html_parts.append(_img_tag(_wordcloud_chart(df["Claim Description"])))

            # Predictions table (first 50 rows)
            display_df = df[["Claim Description", "Predicted Coverage Code"] +
                            ([c for c in ["Accident Source","Coverage Code"] if c in df.columns])].head(50)

            table_html = display_df.to_html(
                index=False, border=0, classes="pred-table",
                escape=True
            )
            html_parts.append(f"""
            <h3 style='margin-top:20px'>📋 Predictions Table (first {len(display_df)} rows)</h3>
            <div style="overflow-x:auto">{table_html}</div>""")

            # Download link for full CSV
            csv_bytes = df.to_csv(index=False).encode()
            csv_b64   = base64.b64encode(csv_bytes).decode()
            html_parts.append(f"""
            <div style="margin-top:16px">
              <a href="data:text/csv;base64,{csv_b64}" download="predictions.csv"
                 style="background:#1976d2;color:white;padding:10px 20px;
                        border-radius:6px;text-decoration:none;font-weight:bold">
                ⬇ Download Full Predictions CSV
              </a>
            </div>""")

            return "\n".join(html_parts)

        except Exception as e:
            logger.exception("Predict pipeline failed")
            return f"<p style='color:red'><b>Error:</b> {str(e)}</p>"
