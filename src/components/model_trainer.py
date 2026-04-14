"""
Model Trainer Component
-----------------------
Trains multiple classifiers on the TF-IDF features and picks the
best-performing one based on test accuracy.

Models evaluated:
  - Logistic Regression
  - Linear SVM (SVC)
  - Multinomial Naive Bayes
  - Random Forest
  - Gradient Boosting (SGD-based, fast for sparse matrices)

Saves the best model + a full report to artifacts/.
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    model_path:  str = os.path.join("artifacts", "best_model.pkl")
    report_path: str = os.path.join("artifacts", "model_report.txt")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_models(self, X_train, X_test, y_train, y_test, label_classes):
        """
        Trains all candidate models and returns a dict of {name: accuracy}.
        """
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=5.0, solver="lbfgs"
            ),
            "Linear SVM": LinearSVC(
                C=1.0, max_iter=2000
            ),
            "Multinomial NB": MultinomialNB(alpha=0.1),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            ),
            "SGD Classifier": SGDClassifier(
                loss="modified_huber", max_iter=200,
                random_state=42, n_jobs=-1
            ),
        }

        results = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc   = accuracy_score(y_test, preds)
                results[name] = (model, acc, preds)
                logger.info(f"  {name}: accuracy = {acc:.4f}")
            except Exception as e:
                logger.warning(f"  {name} failed: {e}")

        return results

    def initialize_model_training(self, X_train, X_test, y_train, y_test, label_classes):
        """
        Runs all models, saves the best one, writes the evaluation report.
        Returns (best_model_name, best_accuracy).
        """
        logger.info("=== Model Training Started ===")
        try:
            results = self.evaluate_models(X_train, X_test, y_train, y_test, label_classes)

            # ── Pick best ─────────────────────────────────────────────────
            best_name = max(results, key=lambda k: results[k][1])
            best_model, best_acc, best_preds = results[best_name]

            logger.info(f"Best model: {best_name}  (accuracy={best_acc:.4f})")

            if best_acc < 0.50:
                raise ValueError(
                    f"No model reached 50% accuracy. Best was {best_acc:.2%}. "
                    "Check dataset size or class balance."
                )

            # ── Save best model ───────────────────────────────────────────
            save_object(self.config.model_path, best_model)

            # ── Write report ──────────────────────────────────────────────
            report_lines = [
                "=" * 60,
                "  INSURANCE CLAIM CATEGORISATION — MODEL REPORT",
                "=" * 60,
                "",
                "  All Models (Test Accuracy):",
            ]
            for name, (_, acc, _) in sorted(results.items(), key=lambda x: -x[1][1]):
                marker = "  *** BEST ***" if name == best_name else ""
                report_lines.append(f"    {name:<25} {acc:.4f}{marker}")

            report_lines += [
                "",
                f"  Best Model : {best_name}",
                f"  Accuracy   : {best_acc:.4f}",
                "",
                "  Classification Report:",
                classification_report(y_test, best_preds, target_names=label_classes),
            ]

            report_text = "\n".join(report_lines)
            with open(self.config.report_path, "w") as f:
                f.write(report_text)

            logger.info("Model report saved")
            logger.info("=== Model Training Complete ===")

            return best_name, best_acc

        except Exception as e:
            raise CustomException(e, sys)
