# Iris Flower Classification 🌸

A complete end-to-end machine learning project that classifies iris flowers into 3 species (setosa, versicolor, virginica) using classic ML algorithms.

## 📊 Dataset
- 150 samples, 4 features, 3 classes
- Source: `sklearn.datasets.load_iris`

## 🚀 Setup
```
git clone <your-repo-url>
cd iris-flower-classification
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🏃 Usage

- Train the model
```bash
python scripts/run_training.py
```

- Run inference
```bash
python scripts/run_inference.py --features 5.1 3.5 1.4 0.2
```

- Run tests
```bash
pytest tests/ -v
```

- Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

## 📈 Results
| Model | Accuracy |
|-------|----------|
| SVM (RBF) | ~96% |
| Logistic Regression | ~93% |
| KNN | ~93% |
| Random Forest | ~90% |

## 📂 Structure

```
iris-flower-classification/
│
├── README.md                     # Project overview, setup, usage
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files to exclude from Git
├── LICENSE                       # Optional: MIT, Apache, etc.
├── setup.py                      # Optional: make project pip-installable
│
├── data/
│   ├── raw/                      # Original, immutable data
│   │   └── iris.csv
│   ├── processed/                # Cleaned/transformed data
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── external/                 # Third-party data (if any)
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb    # Cleaning, scaling, splitting
│   ├── 03_modeling.ipynb         # Training multiple models
│   └── 04_evaluation.ipynb       # Metrics, confusion matrix
│
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                 # Paths, constants, hyperparams
│   ├── data_loader.py            # Load raw data
│   ├── preprocess.py             # Scaling, splitting
│   ├── train.py                  # Model training logic
│   ├── evaluate.py               # Metrics & reports
│   ├── predict.py                # Inference on new data
│   └── utils.py                  # Helper functions
│
├── models/                       # Saved model artifacts
│   ├── iris_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── reports/
│   ├── figures/                  # Plots, charts
│   │   ├── pairplot.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   └── metrics.json              # Final evaluation results
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_predict.py
│
├── api/                          # Optional: deployment
│   ├── app.py                    # Flask/FastAPI app
│   └── schemas.py                # Request/response models
│
├── app/                          # Optional: Streamlit UI
│   └── streamlit_app.py
│
├── configs/                      # Experiment configs (YAML)
│   └── config.yaml
│
└── scripts/                      # CLI entry points
    ├── run_training.py
    └── run_inference.py
```


## 📝 License
MIT License — see LICENSE for details.


