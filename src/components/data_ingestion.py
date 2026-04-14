"""
Data Ingestion Component
------------------------
Reads the raw dataset (CSV or Excel), performs a train/test split,
and saves both splits to the artifacts/ folder.

Dataset columns expected:
  - "Claim Description"  : free-text claim description
  - "Coverage Code"      : target label (e.g. WC, AUTO, GL, PROP)
  - "Accident Source"    : secondary label (optional, kept for EDA)
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    # Paths where processed splits will be saved
    raw_data_path: str   = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str  = os.path.join("artifacts", "test_data.csv")
    source_data_path: str = os.path.join("artifacts", "insurance_claims_data.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        """
        Reads raw data → saves raw copy → splits → saves train & test.
        Returns (train_path, test_path).
        """
        logger.info("=== Data Ingestion Started ===")
        try:
            # ── Read source file ──────────────────────────────────────────
            src = self.config.source_data_path
            if src.endswith(".xlsx") or src.endswith(".xls"):
                df = pd.read_excel(src)
            else:
                df = pd.read_csv(src)

            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # ── Basic validation ──────────────────────────────────────────
            required_cols = ["Claim Description", "Coverage Code"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Drop rows where text or label is null
            df = df.dropna(subset=required_cols)
            df["Claim Description"] = df["Claim Description"].astype(str).str.strip()
            df["Coverage Code"]     = df["Coverage Code"].astype(str).str.strip()

            logger.info(f"After cleaning: {df.shape[0]} valid rows")

            # ── Save raw copy ─────────────────────────────────────────────
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            # ── Train / test split ────────────────────────────────────────
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42,
                stratify=df["Coverage Code"]
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
            logger.info("=== Data Ingestion Complete ===")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
