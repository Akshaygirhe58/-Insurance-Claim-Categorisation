# 🏥 Automated Insurance Claim Categorisation using NLP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br/>

> **Automatically categorise insurance claims from free-text descriptions using Machine Learning and NLP — no cloud, no API keys, runs 100% locally.**

<br/>

![Demo Screenshot](https://i.imgur.com/fGCUcMPl.png)

</div>

---

## 📌 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Coverage Codes](#-coverage-codes)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset Format](#-dataset-format)
- [Models & Performance](#-models--performance)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 About

Insurance companies receive thousands of claim descriptions every day. Manually assigning the correct **Coverage Code** to each claim is slow, error-prone, and expensive.

This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically categorise insurance claims based on their text description. A billing team member uploads a CSV or Excel file of claims, and the system predicts the coverage code for each one — in seconds.

**Business Impact:**

| Metric | Manual Process | This System |
|---|---|---|
| Processing Speed | 3–5 days | Seconds ⚡ |
| Human Error Rate | ~10–15% | Consistent ML output ✅ |
| Scalability | Limited by headcount | Unlimited 🚀 |

---

## ✨ Features

- 📂 **Upload CSV or Excel** files directly in the browser
- 🤖 **5 ML classifiers** trained and compared automatically — best one is selected
- 📊 **Visual analytics** — coverage code distribution, accident source breakdown, word frequency charts
- ⬇️ **Download predictions** as a CSV with one click
- 🔤 **NLP preprocessing pipeline** — stop-word removal, TF-IDF vectorisation (unigrams + bigrams)
- 💾 **Saved model artifacts** — train once, predict instantly every time after
- 📓 **Jupyter notebook** for full Exploratory Data Analysis (EDA)
- 🪵 **Logging & error handling** throughout the pipeline

---

## 📁 Project Structure

```
insurance_claim_NLP/
│
├── 📄 main.py                          ← RUN FIRST — full training pipeline
├── 📄 app.py                           ← RUN SECOND — Flask web application
├── 📄 requirements.txt                 ← Python dependencies
│
├── 📂 artifacts/
│   ├── insurance_claims_data.csv       ← Training dataset
│   ├── best_model.pkl                  ← Saved best model (auto-generated)
│   ├── tfidf_vectorizer.pkl            ← Saved TF-IDF vectorizer (auto-generated)
│   ├── label_encoder.pkl               ← Saved label encoder (auto-generated)
│   └── model_report.txt                ← Accuracy report (auto-generated)
│
├── 📂 src/
│   ├── logger.py                       ← Centralised logging
│   ├── exception.py                    ← Custom exception handler
│   ├── utils.py                        ← Save/load model helpers
│   │
│   ├── 📂 components/
│   │   ├── data_ingestion.py           ← Step 1: Load CSV, train/test split
│   │   ├── data_transformation.py      ← Step 2: NLP cleaning + TF-IDF
│   │   └── model_trainer.py            ← Step 3: Train & evaluate classifiers
│   │
│   └── 📂 pipeline/
│       └── predict_pipeline.py         ← Prediction + chart generation for app
│
├── 📂 templates/
│   └── index.html                      ← Flask frontend (drag & drop UI)
│
└── 📂 notebooks/
    └── EDA.ipynb                       ← 12-section Exploratory Data Analysis
```

---

## ⚙️ How It Works

```
                    ┌─────────────────────────────────────┐
                    │         TRAINING PIPELINE            │
                    │         (run main.py once)           │
                    └─────────────────────────────────────┘

 Raw CSV Dataset
      │
      ▼
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Data Ingestion │────▶│ Data Transformation  │────▶│   Model Training     │
│                 │     │                      │     │                      │
│ • Read CSV      │     │ • Lowercase text     │     │ • Logistic Regression│
│ • Validate cols │     │ • Remove punctuation │     │ • Linear SVM         │
│ • 80/20 split   │     │ • Remove stop-words  │     │ • Multinomial NB     │
│ • Save splits   │     │ • TF-IDF (1,2)-grams │     │ • Random Forest      │
└─────────────────┘     │ • Encode labels      │     │ • SGD Classifier     │
                        │ • Save artifacts     │     │ • Pick best model    │
                        └──────────────────────┘     │ • Save + report      │
                                                      └──────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         PREDICTION PIPELINE          │
                    │         (via web app)                │
                    └─────────────────────────────────────┘

  User uploads CSV
        │
        ▼
  Clean text  ──▶  Load saved TF-IDF  ──▶  Transform  ──▶  Load saved model
                                                                    │
                                                                    ▼
                                               Predict Coverage Code for each claim
                                                                    │
                                                                    ▼
                                              Charts + Table + Downloadable CSV
```

---

## 🏷️ Coverage Codes

| Code | Full Name | Example Claims |
|---|---|---|
| **WC** | Workers' Compensation | Injuries on the job, repetitive strain, machinery accidents |
| **AUTO** | Automobile / Vehicle | Car collisions, vehicle theft, weather-related incidents |
| **GL** | General Liability | Customer injuries on premises, food incidents, assaults |
| **PROP** | Property | Fire, flood, storm damage, theft of property |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/your-username/insurance_claim_NLP.git
cd insurance_claim_NLP
```

**2. Create a virtual environment** *(recommended)*
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the model
```bash
python main.py
```

Expected output:
```
============================================================
  INSURANCE CLAIM NLP — TRAINING PIPELINE
============================================================

[ Step 1/3 ]  Data Ingestion...
  ✓ Train data : artifacts/train_data.csv
  ✓ Test data  : artifacts/test_data.csv

[ Step 2/3 ]  NLP Preprocessing + TF-IDF...
  ✓ TF-IDF features: 416
  ✓ Classes : ['AUTO', 'GL', 'PROP', 'WC']

[ Step 3/3 ]  Training classifiers...

============================================================
  ✅ Training Complete!
  Best Model : Logistic Regression
  Accuracy   : 90.00%
  Report     : artifacts/model_report.txt
============================================================
```

### Step 2 — Start the web app
```bash
python app.py
```
Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Step 3 — Make predictions
1. Upload a CSV or Excel file with a **`Claim Description`** column
2. Click **Run Predictions**
3. View the generated charts and predictions table
4. Click **Download Predictions CSV** to save results

### Step 4 *(Optional)* — Explore the EDA notebook
```bash
cd notebooks
jupyter notebook EDA.ipynb
```

---

## 📊 Dataset Format

### Required column

| Column | Type | Description |
|---|---|---|
| `Claim Description` | Text | Free-text description of the insurance claim |

### Optional columns

| Column | Type | Description |
|---|---|---|
| `Coverage Code` | Text | Actual label — used for training and comparison charts |
| `Accident Source` | Text | Type of accident — generates an additional breakdown chart |

### Example

```csv
Claim Description,Coverage Code,Accident Source
Worker slipped on wet floor and fractured wrist,WC,Slip/Fall
Car rear-ended at traffic light causing whiplash,AUTO,Collision
Customer tripped on loose carpet in hotel lobby,GL,Trip/Fall
Kitchen fire caused by electrical fault,PROP,Fire
```

---

## 🤖 Models & Performance

Five classifiers are trained and evaluated automatically on every run:

| Model | Strengths | Typical Accuracy |
|---|---|---|
| **Logistic Regression** | Fast, interpretable, excellent for sparse text | ⭐ Usually best |
| **Linear SVM** | Great margin-based separation, strong on TF-IDF | ⭐ Usually best |
| **Multinomial NB** | Very fast, solid baseline for text | Good |
| **Random Forest** | Robust, handles non-linearity | Moderate |
| **SGD Classifier** | Fastest training, scalable to large datasets | Good |

> The best model is saved automatically and used for all future predictions.

**NLP Pipeline:**
- Text lowercasing
- Punctuation and number removal
- Stop-word removal (custom list, no NLTK needed)
- TF-IDF Vectorisation with unigrams + bigrams (`ngram_range=(1,2)`)
- Up to 5,000 features with `sublinear_tf=True` scaling

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Web Framework** | Flask |
| **ML Library** | scikit-learn |
| **NLP** | TF-IDF (scikit-learn), custom stop-word removal |
| **Data Processing** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn |
| **Serialisation** | dill / pickle |
| **Excel Support** | openpyxl |
| **Notebook** | Jupyter |

---

## 🐛 Troubleshooting

| Error | Fix |
|---|---|
| `FileNotFoundError: best_model.pkl` | Run `python main.py` first to train the model |
| `KeyError: 'Claim Description'` | Ensure your CSV column is named exactly `Claim Description` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Port 5000 already in use` | Change port in `app.py`: `app.run(port=5001)` |
| Low accuracy | Add more rows to `artifacts/insurance_claims_data.csv` and retrain |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

### Ideas for contributions
- Add more coverage code categories
- Integrate BERT/transformer-based classification
- Add confidence scores to predictions
- Support additional file formats (JSON, XML)
- Add user authentication to the Flask app

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👤 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
- Email: your-email@example.com

---

## 🙏 Acknowledgements

- Original project concept by [siddharthsky](https://github.com/siddharthsky/insurance_claim_NL-p)
- [scikit-learn](https://scikit-learn.org/) — ML library
- [Flask](https://flask.palletsprojects.com/) — Web framework

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>
