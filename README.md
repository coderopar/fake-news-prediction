# Fake News Detection Project

This repository contains the implementation of a machine learning project for **fake news detection** using Logistic Regression, LSTM, and BERT models. It also includes deployment (Streamlit) and visualization (Tableau).

## Repository Structure

```text
fake-news-detection/
│
├── data/                         # (Keep this in .gitignore if large)
│   # Store datasets here (raw, interim, processed)
│
├── notebooks/                    # Jupyter/Colab exploratory notebooks
│   └── fake-news-detection.ipynb # Main notebook for data prep and modelling
│
├── deployment/                   # Streamlit app
│   ├── app.py                    # Main Streamlit script
│   └── requirements.txt          # Dependencies for deployment
│
├── dashboard/                    # Tableau dashboard exports
│   ├── tableau_workbook.twb      # Tableau workbook file
│   └── tableau_screenshots/      # Exported screenshots of visuals
│
├── presentation/                 # Reports, presentations, documents
│   └── presentation.pptx         # Agile workflow slides and results
│
├── .gitignore                    # Ignore data/, cache, artifacts
├── requirements.txt              # Project dependencies
├── README.md                     # Project overview and repo guide
└── LICENSE                       # License information (if open-sourced)
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```