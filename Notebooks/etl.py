import pandas as pd
import os

def run_etl():
    # Build path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '../data/ai_job_dataset.csv')
    output_path = os.path.join(script_dir, '../data/cleaned_ai_jobs.csv')
    
    # 1. Load data
    df = pd.read_csv(input_path)
    
    # 2. Select important analytical features
    required_columns = [
        "job_id", "job_title", "salary_usd", "experience_level", 
        "employment_type", "company_location", "company_size", 
        "remote_ratio", "required_skills", "education_required", 
        "years_experience", "industry", "posting_date"
    ]
    df = df[required_columns]
    
    # 3. Clean the dataset
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna("Unknown")
    
    # Ensure salary_usd is numeric
    df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")
    
    # Convert posting_date to datetime
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    
    # Normalize required_skills to lowercase
    df["required_skills"] = df["required_skills"].str.lower()
    
    # 4. Save cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    run_etl()