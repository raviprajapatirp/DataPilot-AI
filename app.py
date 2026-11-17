import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests

st.set_page_config(page_title="DataPilot AI", page_icon="📊", layout="wide")

st.title("🚀 DataPilot AI")
st.caption("AI-powered data analysis, visualization and machine learning platform")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", [
        "Home", "Upload & Explore", "Data Cleaning",
        "Visualization", "ML Prediction", "AI Assistant", "SQL Helper"
    ])
    st.divider()
    st.caption("Upload a dataset to begin.")

def get_df():
    return st.session_state.get("dataframe")

def read_uploaded_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

if page == "Home":
    st.subheader("Your data analysis workspace")
    st.write("Upload a CSV or Excel file to inspect, clean, visualize and analyze it.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Platform", "DataPilot AI")
    c2.metric("Core", "Python + Streamlit")
    c3.metric("ML", "Scikit-learn")
    st.info("Start in **Upload & Explore**. Train your model using the Jupyter notebook in the notebooks folder.")

elif page == "Upload & Explore":
    st.header("📁 Upload and explore data")
    file = st.file_uploader("Choose CSV or Excel", type=["csv", "xlsx", "xls"])
    if file:
        try:
            df = read_uploaded_file(file)
            st.session_state["dataframe"] = df
            st.success("Dataset loaded.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    df = get_df()
    if df is not None:
        a, b, c = st.columns(3)
        a.metric("Rows", f"{df.shape[0]:,}")
        b.metric("Columns", df.shape[1])
        c.metric("Missing values", int(df.isna().sum().sum()))
        st.subheader("Preview")
        st.dataframe(df.head(100), use_container_width=True)
        st.subheader("Column information")
        info = pd.DataFrame({
            "column": df.columns.astype(str),
            "dtype": df.dtypes.astype(str).values,
            "missing": df.isna().sum().values,
            "unique": df.nunique(dropna=False).values
        })
        st.dataframe(info, use_container_width=True)
        with st.expander("Statistical summary"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

elif page == "Data Cleaning":
    st.header("🧹 Data cleaning")
    df = get_df()
    if df is None:
        st.warning("Upload a dataset first.")
    else:
        st.write("Missing values by column")
        st.dataframe(df.isna().sum().to_frame("missing values"))
        st.write("Duplicate rows:", int(df.duplicated().sum()))
        if st.button("Remove duplicate rows"):
            df = df.drop_duplicates().copy()
            st.session_state["dataframe"] = df
            st.success("Duplicates removed.")
        if st.button("Fill missing values"):
            df = df.copy()
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].notna().any():
                        df[col] = df[col].fillna(df[col].median())
                else:
                    modes = df[col].mode(dropna=True)
                    if not modes.empty:
                        df[col] = df[col].fillna(modes.iloc[0])
            st.session_state["dataframe"] = df
            st.success("Missing values filled where a valid value was available.")
        df = get_df()
        st.dataframe(df.head(100), use_container_width=True)
        st.download_button("Download cleaned CSV",
            df.to_csv(index=False).encode("utf-8"),
            "cleaned_dataset.csv", "text/csv")

elif page == "Visualization":
    st.header("📈 Visualizations")
    df = get_df()
    if df is None:
        st.warning("Upload a dataset first.")
    else:
        cols = df.columns.tolist()
        if not cols:
            st.warning("No columns available.")
        else:
            chart = st.selectbox("Chart type", ["Histogram", "Bar chart", "Scatter plot", "Box plot"])
            if chart == "Histogram":
                col = st.selectbox("Numeric column", df.select_dtypes(include=np.number).columns.tolist())
                st.plotly_chart(px.histogram(df, x=col, title=f"Distribution of {col}"), use_container_width=True)
            elif chart == "Bar chart":
                col = st.selectbox("Column", cols)
                counts = df[col].astype(str).value_counts().head(25).reset_index()
                counts.columns = [col, "count"]
                st.plotly_chart(px.bar(counts, x=col, y="count", title=f"Top values: {col}"), use_container_width=True)
            elif chart == "Scatter plot":
                numeric = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric) < 2:
                    st.warning("Scatter plot needs at least two numeric columns.")
                else:
                    x = st.selectbox("X axis", numeric)
                    y = st.selectbox("Y axis", [v for v in numeric if v != x])
                    st.plotly_chart(px.scatter(df, x=x, y=y), use_container_width=True)
            else:
                numeric = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric:
                    st.warning("Box plot needs a numeric column.")
                else:
                    col = st.selectbox("Numeric column", numeric)
                    st.plotly_chart(px.box(df, y=col), use_container_width=True)

elif page == "ML Prediction":
    st.header("🤖 Prediction using your trained model")
    st.write("Train and export a model in the Jupyter notebook first. Then upload both artifact files.")
    model_file = st.file_uploader("Upload model.joblib", type=["joblib", "pkl"])
    info_file = st.file_uploader("Upload model_info.json", type=["json"])
    new_file = st.file_uploader("Upload new data (CSV/XLSX)", type=["csv", "xlsx", "xls"])
    if model_file and info_file and new_file:
        try:
            model = joblib.load(model_file)
            info = json.load(info_file)
            new_df = read_uploaded_file(new_file)
            features = info["feature_columns"]
            missing = [c for c in features if c not in new_df.columns]
            if missing:
                st.error(f"New data is missing required feature columns: {missing}")
            else:
                predictions = model.predict(new_df[features])
                output = new_df.copy()
                output["prediction"] = predictions
                if info.get("task") == "classification" and hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(new_df[features])
                    output["prediction_confidence"] = probabilities.max(axis=1)
                st.success(f"Predictions generated. Model: {info.get('model_name', 'Unknown')}")
                st.dataframe(output, use_container_width=True)
                st.download_button("Download predictions",
                    output.to_csv(index=False).encode("utf-8"),
                    "datapilot_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "AI Assistant":
    st.header("💬 AI data assistant (Ollama)")
    st.caption("Start Ollama locally and pull a model before using this feature.")
    model_name = st.text_input("Ollama model name", "llama3:latest")
    question = st.text_area("Ask a question about the uploaded dataset")
    df = get_df()
    if st.button("Ask AI"):
        if df is None:
            st.warning("Upload a dataset first.")
        elif not question.strip():
            st.warning("Enter a question.")
        else:
            summary = df.head(20).to_csv(index=False)
            prompt = (
                "You are a careful data analyst. Answer using only the supplied "
                "dataset preview and summary. If the information is insufficient, say so. "
                "Do not claim to have calculated values you cannot verify.\n\n"
                f"Shape: {df.shape}\nColumns: {df.dtypes.astype(str).to_dict()}\n"
                f"Missing values: {df.isna().sum().to_dict()}\n"
                f"Preview:\n{summary}\n\nQuestion: {question}"
            )
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                    timeout=120
                )
                response.raise_for_status()
                st.write(response.json().get("response", "No response returned."))
            except Exception as e:
                st.error(f"Could not connect to Ollama: {e}. Check that Ollama is running and the model is installed.")

elif page == "SQL Helper":
    st.header("🗄️ Natural language to SQL helper")
    st.warning("Generated SQL is displayed for review only. It is not executed against a database.")
    model_name = st.text_input("Ollama model name", "llama3:latest", key="sql_model")
    dialect = st.selectbox("SQL dialect", ["MySQL", "SQLite", "PostgreSQL"])
    schema = st.text_area("Describe your database tables and columns", "Table: customers\nColumns: customer_id, age, city, monthly_charges")
    question = st.text_area("What SQL do you need?", "Show the average monthly_charges by city.")
    if st.button("Generate SQL"):
        prompt = (
            f"Write one safe, read-only {dialect} SELECT query only. "
            "Use only the schema supplied. Do not invent tables or columns. "
            "If the schema is insufficient, explain what is missing.\n\n"
            f"Schema:\n{schema}\n\nRequest:\n{question}"
        )
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=120
            )
            response.raise_for_status()
            st.code(response.json().get("response", ""), language="sql")
        except Exception as e:
            st.error(f"Could not connect to Ollama: {e}")

st.divider()
st.caption("DataPilot AI • Educational portfolio project • Validate outputs before making business decisions.")
