# Sentiment Analysis

An end-to-end sentiment analysis project where multiple machine learning and deep learning models were implemented, compared, and evaluated.  
The best-performing model (Artificial Neural Network) was deployed as a production-ready web application using Streamlit.

---

## Project Summary

This project performs multi-class sentiment classification on text data.  
Three different models were trained and compared:

- Logistic Regression (TF-IDF features)
- Random Forest (TF-IDF features)
- Artificial Neural Network (Embedding + Dense layers)

After evaluation, the ANN model achieved the highest validation performance and was selected for deployment.

The model classifies text into four categories:

- Positive  
- Negative  
- Neutral  
- Irrelevant  

---

## Project Workflow

### Data Preprocessing
- Lowercasing
- Punctuation removal
- Tokenization
- TF-IDF vectorization (for traditional ML)
- Sequence padding (for ANN)

The same preprocessing pipeline used during training is applied during deployment to ensure consistency.

---

### Model Training & Comparison

| Model | Feature Representation |
|--------|------------------------|
| Logistic Regression | TF-IDF |
| Random Forest | TF-IDF |
| ANN | Tokenized Sequences + Embedding |

All models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Validation Loss

---

## Final Model Performance (ANN)

- Training Accuracy: ~96.5%
- Validation Accuracy: ~96.1%
- Validation Loss: ~0.19

The small gap between training and validation accuracy indicates good generalization and minimal overfitting.

---

## ANN Architecture

- Embedding Layer
- Dense Hidden Layers
- Softmax Output Layer (4 classes)

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam  
**Evaluation Metric:** Accuracy  

The model outputs probability scores for each sentiment class, and the highest probability determines the final prediction.

---

## Web Application Features

- Real-time sentiment prediction
- Confidence score display
- Probability distribution visualization
- Model performance section
- Technical explanation section

---

## Repository Structure

```
sentiment-analysis-/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   └── sentiment.ipynb
│
├── src/
│   ├── data/
│   │   └── data_loader.py
│   │
│   ├── features/
│   │   └── preprocessor.py
│   │
│   ├── models/
│   │   ├── trainer.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   ├── pipeline/
│   │   └── pipeline.py
│   │
│   └── utils/
│       ├── logger.py
│       └── config_loader.py
│
├── configs/
│   └── config.yaml
│
├── artifacts/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── ui.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/psawner/sentiment-system-
cd Sentiment-Analysis
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Streamlit
- AWS (Deployment)

---

## Limitations

- May struggle with sarcasm or context-heavy text
- Short inputs may produce lower confidence
- Performance depends on dataset distribution

---

## Key Highlights

- Compared traditional ML and deep learning approaches
- Proper validation and evaluation methodology
- Selected best-performing model based on metrics
- Successfully deployed ANN model as an interactive web app