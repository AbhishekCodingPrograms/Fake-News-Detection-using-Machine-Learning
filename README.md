# Fake News Detection using Machine Learning
LIVE : https://fake-news-detection-using-machine-learning-abhishek.streamlit.app/
A Streamlit web application that classifies a news article as **fake**, **real**, or **uncertain** using predictions from Logistic Regression and Random Forest models.

## Features

- Simple professional dark-mode interface
- TF-IDF text vectorization
- Logistic Regression prediction
- Random Forest prediction
- Averaged ensemble probability
- Fake and real probability display
- Model agreement display
- Uncertain result when neither class reaches 60% confidence

## Requirements

- Python 3.9 or newer
- The packages listed in [`requirements.txt`](./requirements.txt)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AbhishekCodingPrograms/Fake-News-Detection-using-Machine-Learning.git
   cd Fake-News-Detection-using-Machine-Learning
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the application

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

Paste a headline and article body into the text area, then select **Detect article**.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── vectorization.pkl
├── lr_model.pkl
└── rf_model.pkl
```

- `app.py`: Streamlit application
- `vectorization.pkl`: trained TF-IDF vectorizer
- `lr_model.pkl`: trained Logistic Regression model
- `rf_model.pkl`: trained Random Forest model

## Important note

This application identifies patterns learned from its training data. Its output is a machine-learning classification and is **not an independent fact check**. Always verify important claims with reliable sources.

## Model files

The Random Forest model is larger than GitHub's recommended 50 MB file size. Git LFS is recommended if the model files need to be updated or expanded in future.
