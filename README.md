# DataPilot AI

An AI-powered data analysis and machine-learning portfolio project built with Python, Jupyter Notebook, Streamlit, scikit-learn and Ollama.

## Features
- Upload CSV and Excel datasets
- Inspect data types, missing values, duplicates and descriptive statistics
- Clean duplicates and missing values
- Interactive histogram, bar, scatter and box plots
- Train a classification or regression model in Jupyter Notebook
- Export a scikit-learn pipeline and use it to predict on new data
- Optional local AI data assistant and natural-language-to-SQL helper using Ollama

## Project structure
```text
DataPilot-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── notebooks/
│   └── DataPilot_AI_Model_Training.ipynb
├── artifacts/
└── data/
```

## 1. Setup
Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```
Activate it:
- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install packages:
```bash
pip install -r requirements.txt
```

## 2. Train a model in Jupyter
```bash
jupyter notebook
```
Open `notebooks/DataPilot_AI_Model_Training.ipynb`.
Put your CSV/XLSX dataset in `data/`, update `DATA_PATH`, `TARGET_COLUMN` and `TASK`, then run all cells.

The notebook exports:
- `artifacts/model.joblib`
- `artifacts/model_info.json`

Upload both files in the Streamlit app's **ML Prediction** page, together with a new CSV/XLSX containing the same input feature columns.

## 3. Run Streamlit
From the project root:
```bash
streamlit run app.py
```

## 4. Optional Ollama setup
Install Ollama separately, start it, and download a model, for example:
```bash
ollama pull llama3:latest
```
Keep Ollama running locally. In the app, use `llama3:latest` as the model name in the app (or the exact name shown by `ollama list`). The AI features call the local Ollama API at `http://localhost:11434`.

**Deployment note:** Streamlit Community Cloud cannot access Ollama running on your personal computer. The Ollama features work locally unless you configure a securely hosted model endpoint. Never expose Ollama publicly without appropriate security.

## 5. Push to GitHub
Create a new empty GitHub repository named `DataPilot-AI`, then run:
```bash
git init
git add .
git commit -m "Initial DataPilot AI project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/DataPilot-AI.git
git push -u origin main
```

## Important notes
- Do not commit private, confidential or personally identifiable data.
- Do not commit API keys, passwords, `.env`, or Streamlit secrets.
- Review model metrics and validate predictions on genuinely unseen data.
- Random Forest is a baseline model, not proof that a model is suitable for production.
- The SQL helper only generates text; it does not connect to or execute queries on MySQL.
- The AI assistant receives a small dataset preview and metadata, not the entire dataset.
