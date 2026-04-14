"""
main.py  —  Run this FIRST before starting the web app.
--------------------------------------------------------------
This script runs the full training pipeline:
  Step 1: Data Ingestion    → reads dataset, splits train/test
  Step 2: Data Transformation → NLP cleaning + TF-IDF vectorisation
  Step 3: Model Training    → trains 5 classifiers, saves the best one

Usage:
    python main.py

After this runs successfully, you will see:
  artifacts/best_model.pkl
  artifacts/tfidf_vectorizer.pkl
  artifacts/label_encoder.pkl
  artifacts/model_report.txt

Then run:  python app.py
"""

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  INSURANCE CLAIM NLP — TRAINING PIPELINE")
    print("="*60 + "\n")

    # ── Step 1: Data Ingestion ─────────────────────────────────────
    print("[ Step 1/3 ]  Data Ingestion...")
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initialize_data_ingestion()
    print(f"  ✓ Train data : {train_path}")
    print(f"  ✓ Test data  : {test_path}\n")

    # ── Step 2: Data Transformation ────────────────────────────────
    print("[ Step 2/3 ]  NLP Preprocessing + TF-IDF...")
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, label_classes = \
        transformer.initialize_data_transformation(train_path, test_path)
    print(f"  ✓ TF-IDF features: {X_train.shape[1]}")
    print(f"  ✓ Classes : {label_classes}\n")

    # ── Step 3: Model Training ─────────────────────────────────────
    print("[ Step 3/3 ]  Training classifiers...")
    trainer = ModelTrainer()
    best_name, best_acc = trainer.initialize_model_training(
        X_train, X_test, y_train, y_test, label_classes
    )

    print(f"\n{'='*60}")
    print(f"  ✅ Training Complete!")
    print(f"  Best Model : {best_name}")
    print(f"  Accuracy   : {best_acc:.2%}")
    print(f"  Report     : artifacts/model_report.txt")
    print(f"{'='*60}")
    print("\n  Now run:  python app.py")
    print("  Then open:  http://127.0.0.1:5000\n")
