# Fake News Detection Project

## Overview
This project aims to **detect fake news articles** using multiple machine learning and deep learning models.  
It compares traditional models like **Logistic Regression (TF–IDF)** with deep models like **LSTM** and **BERT**, evaluating their accuracy, F1-score, and robustness across **cleaned** and **combined** text datasets.

The system is designed to run both **locally** and on **Kaggle** with GPU acceleration, supporting smooth transitions between environments.

---

## Repository Structure

```
fake-news-detection/
│
├── data/                         # Raw, processed, and split datasets (.gitignored)
│
├── notebooks/
│   └── fake-news-detection.ipynb  # Main notebook: preprocessing, modelling, evaluation
│
├── deployment/
│   ├── app.py                    # Streamlit web app for live fake news detection
│   └── requirements.txt          # Dependencies for deployment environment
│
├── dashboard/
│   └──tableau_screenshots/      # Screenshots or exports of dashboards
│        
│
├── presentation/
│   └── presentation.pptx         # Project slides (workflow, results, insights)
│
├── .gitignore                    # Ignores data/, cache, and large artifacts
├── requirements.txt              # Global dependencies for local runs
├── README.md                     # Project documentation (you’re here)
```

---

## Data Source

### Description

This dataset is a collection of news articles labeled as either **“fake”** or **“real”**.  
It is designed for **binary classification** tasks in **natural language processing (NLP)**, especially for **fake news detection**.

It is also referred to as the **ISOT Fake News Detection Dataset**.  
*(Source: [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))*

The dataset comprises **two CSV files** containing the articles and their labels.  
*(Source: [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))*

The total size is about **43 MB**.  
*(Source: [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))*

It is downloaded as one .zip file called `archive.zip`

---

### Contents / Structure

Each record (row) in the dataset typically contains:

| **Column** | **Description** |
|:------------|:----------------|
| `title` | The headline or title of the news article |
| `text` | The full text or body of the news article |
| `subject` | Subject that the article covers |
| `date` | The date when the article was released |

## Environment Setup

### Option 1 — Local setup

```bash
# Clone repo
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

Place your dataset archive (e.g. `archive.zip`) in the `data/` directory before running the notebook.

---

### Option 2 — Run on Kaggle

1. Upload the notebook to Kaggle.  
2. Set **Settings** → **Accelerator** → **GPU T4 x 2**.  
3. Mount dataset under `/kaggle/input/fake-news-dataset/archive.zip`.  
4. The notebook automatically detects the Kaggle environment and sets all paths accordingly (`/kaggle/working/...`).

---

##  Model Pipelines

| Model | Description | Input Type | Notes |
|-------|--------------|------------|-------|
| **Logistic Regression (TF–IDF)** | Classical baseline with vectorized word frequencies. | Clean & combined text | Fast, interpretable baseline. |
| **LSTM (Keras)** | Deep sequential model capturing long-term dependencies. | Tokenized padded sequences | Tuned for accuracy and generalization. |
| **BERT (HuggingFace Transformers)** | Pretrained transformer fine-tuned on fake/real labels. | Clean & combined text | GPU-intensive but highest F1. |

---

## Notebook Flow

1. **Imports and Setup**  
   - Downloads NLTK resources (stopwords, wordnet, punkt).  
   - Detects environment (local/Kaggle) and sets file paths accordingly.

2. **Data Loading and Extraction**  
   - Extracts the dataset archive.  
   - Reads and merges `fake` and `true` CSVs.

3. **Data Understanding**
   - Descriptive Statistics
   - Class Distribution Analysis
   - Exploratory Word Clouds

4. **Text Preprocessing**  
   - Cleaning: lowercasing, removing URLs, punctuation, and stopwords.  
   - Lemmatization: normalizing words.  
   - Combination: merging title + text for the “combined” dataset.

5. **Model Training**  
   - Logistic Regression trained on TF–IDF features.  
   - LSTM model created via custom `create_lstm_model()` function.  
   - BERT fine-tuned using HuggingFace `Trainer` API.

6. **Evaluation**  
   - Accuracy, Precision, Recall, and F1-score computed for each variant.  
   - Stored predictions for Comparison Table

7. **Artifact Saving**  
   - Trained models, vectorizers, and plots saved to `models/` and `results/` subfolders (auto-created).

8. **Comparison Table**
   ```text
   | Model | Variant  | Accuracy | Precision | Recall | F1 |

   ```
---

## Deployment
- A Dash app deployed at `https://capstone-project-f85dcbe34c51.herokuapp.com/`. The link is also in the repository's `About` section

---

## Results Table

| **Model** | **Variant** | **Accuracy** | **Precision** | **Recall** | **F1-score** |
|:-----------|:-------------|:-------------|:--------------|:------------|:--------------|
| **BERT** | Combined | 0.999095 | 0.999095 | 0.999095 | 0.999095 |
| **LSTM** | Clean | 0.998869 | 0.998871 | 0.998869 | 0.998868 |
| **BERT** | Clean | 0.998416 | 0.998417 | 0.998416 | 0.998416 |
| **LSTM** | Combined | 0.998190 | 0.998190 | 0.998190 | 0.998190 |
| **LogReg** | Clean | 0.987101 | 0.987141 | 0.987101 | 0.987103 |
| **LogReg** | Combined | 0.986875 | 0.986958 | 0.986875 | 0.986877 |

---

## Dashboard and Presentation

- **Tableau Dashboard:** Visualizes class distributions, model comparisons, and top words in fake vs real news.  
- **Presentation Slides:** Summarize the project workflow, results, and recommendations for deployment.

---

## Key Highlights

- End-to-end reproducibility (Kaggle or local)  
- Modular directory structure (data → model → deployment)  
- Supports multiple model types (classical, deep learning, transformer)  
- Includes visualization & deployment artifacts  
- Clean, well-documented notebook and codebase  

---

## Future Work

- Integrate **attention visualization** for BERT predictions.  
- Extend to **multilingual fake news** datasets.  
- Deploy on **HuggingFace Spaces** or **AWS Lambda**.  
- Add **explainability (SHAP/LIME)** components.

---

## Contributors
- Steve Opar
- Kelly Kihige
- Kungu Washington
- Olive Njeri
- Mercy Chepkorir
- Esterina Kananu