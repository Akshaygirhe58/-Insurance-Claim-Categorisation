import os
import sys

# Try dill first (handles more object types), fall back to pickle
try:
    import dill as serializer
except ImportError:
    import pickle as serializer

from src.logger import logger
from src.exception import CustomException


def save_object(file_path: str, obj):
    """Serialize any Python object to a file using dill (or pickle as fallback)."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            serializer.dump(obj, f)
        logger.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Deserialize a pickled object from file."""
    try:
        with open(file_path, "rb") as f:
            obj = serializer.load(f)
        logger.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
