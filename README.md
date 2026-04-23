# AI Job Market Analyzer 🚀

A comprehensive end-to-end data engineering and machine learning pipeline that analyzes the AI job market. This project extracts job data, cleans it, trains machine learning models to predict salaries, and visualizes hiring trends using an interactive dashboard.

## 🌟 Features

- **ETL Pipeline**: Robust data cleaning and preprocessing to prepare raw job market data.
- **Machine Learning**: Utilizes Scikit-Learn to train Random Forest and Linear Regression models for accurate salary predictions based on skills and job features.
- **Skill Analysis**: Natural language processing (NLP) to extract and analyze the most demanded skills in the AI industry.
- **Interactive Dashboard**: A beautiful, real-time Streamlit dashboard providing insights into:
  - Hiring trends over time
  - Most demanded AI skills
  - Salary distributions across roles
  - Remote vs. On-site work ratios
- **SQL Analytics**: Pre-defined SQL queries for in-depth database analysis.

## 📂 Project Structure

```
Market_analyzer/
├── Notebooks/
│   └── etl.py              # ETL pipeline script
├── data/
│   ├── ai_job_dataset.csv  # Raw dataset
│   └── cleaned_ai_jobs.csv # Processed dataset
├── dashboard.py            # Streamlit interactive dashboard
├── ml_models.py            # Machine learning pipeline and skill analysis
├── sql_analytics.sql       # SQL queries for market analysis
├── requirements.txt        # Python dependencies
└── instructions.md         # Original execution instructions
```

## ⚙️ Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```
*(Alternatively, you can manually install the main packages: `pip install pandas scikit-learn plotly streamlit`)*

## 🚀 Running the Project

Follow these steps in order to execute the full pipeline:

### 1. Run the ETL Pipeline
The ETL (Extract, Transform, Load) script cleans the raw dataset and prepares it for the ML models and dashboard.

```bash
python Notebooks/etl.py
```
*This reads `data/ai_job_dataset.csv` and generates `data/cleaned_ai_jobs.csv`.*

### 2. Run the Machine Learning Models
This script performs skill analysis, vectorizes required skills, and trains the salary prediction models.

```bash
python ml_models.py
```

### 3. Launch the Interactive Dashboard
Explore hiring trends, most demanded skills, salary distributions, and remote work ratios interactively.

```bash
streamlit run dashboard.py
```
*This starts a local web server (usually at `http://localhost:8501`). Open the provided URL in your browser to view the application.*
