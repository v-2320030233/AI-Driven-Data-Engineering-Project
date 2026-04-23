import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer

def analyze_skills(df):
    """
    STEP 3: SKILL ANALYSIS
    Process the required_skills column to:
    - split skills into tokens
    - compute most frequent skills
    - analyze technology demand
    """
    print("\n--- Skill Analysis ---")
    
    # Filter valid skills
    skills_series = df["required_skills"].copy()
    skills_series = skills_series[skills_series.notna() & (skills_series != "unknown")]
    
    # Vectorize and tokenize skills using CountVectorizer
    # Assuming skills are comma separated
    vectorizer = CountVectorizer(tokenizer=lambda x: [s.strip() for s in str(x).split(',')], lowercase=True, token_pattern=None)
    skill_matrix = vectorizer.fit_transform(skills_series)
    
    # Compute most frequent skills
    skill_counts = pd.DataFrame({
        'skill': vectorizer.get_feature_names_out(),
        'count': skill_matrix.sum(axis=0).A1
    }).sort_values('count', ascending=False)
    
    print("Top 15 most demanded skills:")
    print(skill_counts.head(15))
    return skill_counts

def train_salary_model(df):
    """
    STEP 4: MACHINE LEARNING MODELS
    Build machine learning models using the cleaned dataset to predict salary.
    """
    print("\n--- Training Salary Prediction Model ---")
    
    # Filter dataset for modeling
    model_df = df.copy()
    model_df['salary_usd'] = pd.to_numeric(model_df['salary_usd'], errors='coerce')
    model_df = model_df.dropna(subset=['salary_usd'])
    
    # Target variable and Features
    features = ['experience_level', 'company_size', 'remote_ratio', 'years_experience', 'industry']
    target = 'salary_usd'
    
    X = model_df[features]
    y = model_df[target]
    
    # Simple imputation and encoding for categorical/mixed types
    categorical_features = ['experience_level', 'company_size', 'industry', 'years_experience', 'remote_ratio']
    
    # Preprocessor for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    
    # Modeline pipelines
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Random Forest
    print("Training RandomForestRegressor...")
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    print(f"Random Forest RMSE: {root_mean_squared_error(y_test, rf_preds):.2f}")
    print(f"Random Forest R2: {r2_score(y_test, rf_preds):.4f}")
    
    # Train and evaluate Linear Regression
    print("\nTraining LinearRegression...")
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    print(f"Linear Regression RMSE: {root_mean_squared_error(y_test, lr_preds):.2f}")
    print(f"Linear Regression R2: {r2_score(y_test, lr_preds):.4f}")
    
    return rf_pipeline, lr_pipeline

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'data/cleaned_ai_jobs.csv')
    
    if os.path.exists(input_path):
        cleaned_df = pd.read_csv(input_path)
        analyze_skills(cleaned_df)
        train_salary_model(cleaned_df)
    else:
        print(f"File not found: {input_path}. Please run the ETL script first to generate the cleaned dataset.")
