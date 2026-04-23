# AI Job Market Analyzer - Run Instructions

This document provides the necessary commands to run the entire AI Job Market Analyzer pipeline. The project consists of data generation/extraction (ETL), a machine learning model pipeline, and an interactive Streamlit dashboard.

## Prerequisites
Ensure you have the required Python packages installed. If you haven't installed them yet, you can do so by running:

```bash
pip install pandas scikit-learn plotly streamlit
```

## Running the Pipeline

The project should be executed in the following order:

### 1. Run the ETL Pipeline
The ETL (Extract, Transform, Load) script cleans the raw dataset and prepares it for the machine learning models and dashboard.

Run the following command from the project root:
```bash
python Notebooks/etl.py
```
*This will read `data/ai_job_dataset.csv` and generate `data/cleaned_ai_jobs.csv`.*

### 2. Run the Machine Learning Models
This script performs skill analysis, vectorizes jobs required skills, and trains the salary prediction models (Random Forest and Linear Regression).

Run the following command from the project root:
```bash
python ml_models.py
```

### 3. Launch the Streamlit Dashboard
The dashboard allows you to explore hiring trends, most demanded skills, salary distributions, and remote work ratios interactively.

Run the following command from the project root:
```bash
streamlit run dashboard.py
```

*This will start a local web server (usually at `http://localhost:8501`) where you can view the application in your browser.*
