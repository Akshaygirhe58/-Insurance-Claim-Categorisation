"""
Data Transformation Component
------------------------------
Applies NLP preprocessing to the "Claim Description" column:
  1. Lowercase + remove punctuation / numbers
  2. Tokenise
  3. Remove English stop-words
  4. (Optional) Lemmatization via simple suffix rules — no NLTK downloads needed
  5. TF-IDF vectorisation (unigrams + bigrams, max 5000 features)

Saves the fitted TF-IDF vectoriser and label encoder to artifacts/.
Returns (X_train, X_test, y_train, y_test) as numpy arrays.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

# ── Minimal stop-word list (no NLTK dependency) ───────────────────────────────
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","was","were","are","be","been","has","have","had",
    "it","its","this","that","these","those","i","me","my","we","our",
    "you","your","he","she","his","her","they","their","which","who",
    "what","when","where","how","as","if","so","up","out","about","into",
    "during","after","before","under","over","while","due","caused","due",
}


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, remove stop-words."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters + spaces
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


@dataclass
class DataTransformationConfig:
    vectorizer_path: str    = os.path.join("artifacts", "tfidf_vectorizer.pkl")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initialize_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train & test CSVs, cleans text, fits TF-IDF on train,
        transforms both splits, encodes labels.

        Returns (X_train, X_test, y_train, y_test, label_classes)
        """
        logger.info("=== Data Transformation Started ===")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            # ── Clean text ────────────────────────────────────────────────
            train_df["clean_text"] = train_df["Claim Description"].apply(clean_text)
            test_df["clean_text"]  = test_df["Claim Description"].apply(clean_text)

            logger.info("Text cleaning complete")

            # ── TF-IDF ────────────────────────────────────────────────────
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),     # unigrams + bigrams
                max_features=5000,
                sublinear_tf=True,      # apply log(1+tf) scaling
                min_df=1,
            )

            X_train = vectorizer.fit_transform(train_df["clean_text"])
            X_test  = vectorizer.transform(test_df["clean_text"])

            logger.info(f"TF-IDF matrix: train={X_train.shape}, test={X_test.shape}")

            # ── Label encoding ────────────────────────────────────────────
            le = LabelEncoder()
            y_train = le.fit_transform(train_df["Coverage Code"])
            y_test  = le.transform(test_df["Coverage Code"])

            logger.info(f"Classes: {list(le.classes_)}")

            # ── Save objects ──────────────────────────────────────────────
            save_object(self.config.vectorizer_path, vectorizer)
            save_object(self.config.label_encoder_path, le)

            logger.info("=== Data Transformation Complete ===")

            return X_train, X_test, y_train, y_test, list(le.classes_)

        except Exception as e:
            raise CustomException(e, sys)
