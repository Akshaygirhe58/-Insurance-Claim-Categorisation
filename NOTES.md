# 📚 Insurance Claim NLP — Complete Study Notes

These notes explain every concept and every file in this project, written
for someone learning ML / NLP for the first time.

---

## 🧠 What Does This Project Do?

An insurance company receives thousands of claim descriptions like:
> "Worker fell from scaffold at construction site and fractured his leg."

A human needs to read this and assign it a **Coverage Code** like `WC`
(Workers' Compensation). This project trains a machine learning model
to do that automatically using **NLP** (Natural Language Processing).

---

## 📁 Project Structure (What Each File Does)

```
insurance_claim_NLP/
│
├── main.py                 ← RUN FIRST — trains the model
├── app.py                  ← RUN SECOND — starts the web app
├── requirements.txt        ← list of Python packages to install
│
├── artifacts/
│   ├── insurance_claims_data.csv  ← THE DATASET (your training data)
│   ├── best_model.pkl             ← saved trained model (auto-generated)
│   ├── tfidf_vectorizer.pkl       ← saved text vectorizer (auto-generated)
│   ├── label_encoder.pkl          ← saved label encoder (auto-generated)
│   └── model_report.txt           ← accuracy report (auto-generated)
│
├── src/
│   ├── logger.py           ← logs events to a file for debugging
│   ├── exception.py        ← custom error messages with file + line number
│   ├── utils.py            ← helper to save/load .pkl files
│   │
│   ├── components/
│   │   ├── data_ingestion.py      ← Step 1: read CSV, split train/test
│   │   ├── data_transformation.py ← Step 2: clean text, TF-IDF
│   │   └── model_trainer.py       ← Step 3: train + compare classifiers
│   │
│   └── pipeline/
│       └── predict_pipeline.py    ← used by app.py to make predictions
│
├── templates/
│   └── index.html          ← the HTML page shown in the browser
│
└── notebooks/
    └── EDA.ipynb           ← Exploratory Data Analysis (Jupyter)
```

---

## 🔄 How the Pipeline Works (Step by Step)

```
RAW DATASET (CSV)
      │
      ▼
[ data_ingestion.py ]
  - Reads the CSV
  - Validates required columns
  - Splits into 80% train / 20% test
  - Saves both to artifacts/
      │
      ▼
[ data_transformation.py ]
  - Cleans each claim text:
      lowercase → remove punctuation → remove stop-words
  - Fits TF-IDF vectorizer on training text
  - Converts text → numeric matrix (rows=claims, cols=word features)
  - Encodes labels: "WC" → 3, "AUTO" → 0, etc.
  - Saves vectorizer + encoder to artifacts/
      │
      ▼
[ model_trainer.py ]
  - Trains 5 different classifiers on the TF-IDF matrix
  - Evaluates each on test set
  - Picks the best by accuracy
  - Saves the best model to artifacts/
      │
      ▼
[ app.py + predict_pipeline.py ]
  - User uploads a new CSV via browser
  - Text is cleaned and vectorized using the SAVED vectorizer
  - Model predicts Coverage Code for each claim
  - Charts + table are shown in the browser
```

---

## 📖 Key Concepts Explained

### What is NLP?
Natural Language Processing = teaching computers to understand text.
In this project we convert claim descriptions (text) into numbers
that a machine learning model can work with.

### What is TF-IDF?
**TF-IDF** = Term Frequency × Inverse Document Frequency.

- **TF (Term Frequency)**: How often a word appears in a single claim.
  Example: if "injury" appears 3 times in a 10-word claim → TF = 0.3
- **IDF (Inverse Document Frequency)**: Penalises very common words.
  A word that appears in every single claim (like "the") gets a low IDF.
  A rare word like "asbestos" gets a high IDF.
- **TF-IDF score** = TF × IDF → high score = word is important in THIS
  document but rare across all documents.

Result: a matrix where each row is a claim and each column is a word.
Values represent how important each word is to that claim.

### What are N-grams?
We use `ngram_range=(1, 2)` which means we use:
- **Unigrams**: single words → "slip", "fall", "injury"
- **Bigrams**: pairs of words → "slip fall", "back injury", "car accident"

Bigrams capture context that single words miss.

### What is Stop-Word Removal?
Words like "the", "a", "is", "was" carry no meaning for classification.
We remove them so the model focuses on meaningful words.

### What is a Label Encoder?
Converts string labels to integers:
  `AUTO → 0`, `GL → 1`, `PROP → 2`, `WC → 3`
Machine learning models require numbers, not strings.

### What is Train/Test Split?
We split data into two parts:
- **Train set (80%)**: model learns from this
- **Test set (20%)**: model is evaluated on this (it has never seen these)

This tests whether the model can generalise to new, unseen data.

### What is dill?
`dill` is like Python's `pickle` but more powerful — it can serialise
more types of objects (like custom functions inside scikit-learn pipelines).
We use it to save the trained model to a `.pkl` file and reload it later
without retraining.

---

## 🤖 The 5 Models Compared

| Model | How It Works | Best For |
|---|---|---|
| **Logistic Regression** | Finds a linear boundary between classes using log-odds | Fast, interpretable, great baseline for text |
| **Linear SVM** | Finds the maximum-margin hyperplane separating classes | Excellent for high-dimensional sparse text features |
| **Multinomial NB** | Uses Bayes' theorem with word count probabilities | Very fast, works well with TF-IDF, good for small data |
| **Random Forest** | Ensemble of decision trees voting on the answer | Robust, handles non-linear patterns |
| **SGD Classifier** | Stochastic Gradient Descent — fast online learning | Very fast on large sparse datasets |

For text classification with TF-IDF features, **Linear SVM** and
**Logistic Regression** usually win because the feature space is
high-dimensional and sparse.

---

## 📊 Coverage Codes Explained

| Code | Meaning | Example |
|---|---|---|
| **WC** | Workers' Compensation | Worker injured on the job |
| **AUTO** | Automobile / Vehicle | Car collision, vehicle theft |
| **GL** | General Liability | Customer injured on premises |
| **PROP** | Property | Fire, flood, theft of property |

---

## 🚀 How to Run (Step by Step)

### Step 1 — Set up Python environment
```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model
```bash
python main.py
```
You will see:
```
[ Step 1/3 ]  Data Ingestion...
[ Step 2/3 ]  NLP Preprocessing + TF-IDF...
[ Step 3/3 ]  Training classifiers...
  Logistic Regression: accuracy = 0.9000
  Linear SVM:          accuracy = 0.9500
  ...
Best Model : Linear SVM
Accuracy   : 95.00%
```

### Step 4 — Start the web app
```bash
python app.py
```
Open your browser at: **http://127.0.0.1:5000**

### Step 5 — Use the app
1. Upload the file `artifacts/insurance_claims_data.csv`
   (or make your own CSV with a `Claim Description` column)
2. Click "Run Predictions"
3. View charts + predictions table
4. Download the full predictions CSV

### Step 6 — Explore the EDA notebook
```bash
cd notebooks
jupyter notebook EDA.ipynb
```

---

## 📝 Dataset Format

Your input CSV must have at minimum:

| Column | Required? | Description |
|---|---|---|
| `Claim Description` | ✅ Yes | Free text description of the claim |
| `Coverage Code` | For training only | The correct label (WC, AUTO, GL, PROP) |
| `Accident Source` | Optional | Type of accident (for charts) |

---

## 🐛 Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: best_model.pkl` | You haven't trained yet | Run `python main.py` first |
| `KeyError: Claim Description` | Column name is wrong | Make sure your CSV has exactly `Claim Description` as the column name |
| `ModuleNotFoundError: flask` | Packages not installed | Run `pip install -r requirements.txt` |
| `Port 5000 already in use` | Another app is using port 5000 | Change port in app.py: `app.run(port=5001)` |
| Low accuracy (<60%) | Dataset too small | Add more rows to the dataset |

---

## 💡 What to Learn Next

1. **NLTK / spaCy** — more powerful NLP: lemmatization, POS tagging, NER
2. **Word Embeddings** — Word2Vec, GloVe: represent words as dense vectors
3. **BERT / Transformers** — state-of-the-art NLP models (HuggingFace)
4. **Cross-Validation** — more robust model evaluation than single train/test split
5. **Hyperparameter Tuning** — GridSearchCV to find best model settings
6. **MLflow** — experiment tracking (log every model run)
7. **Docker** — containerise the app so it runs anywhere
