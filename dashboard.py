import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="AI Job Market Analytics", layout="wide")

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/cleaned_ai_jobs.csv")
    df = pd.read_csv(data_path)
    df['salary_usd'] = pd.to_numeric(df['salary_usd'], errors='coerce')
    df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
    return df

st.title("📊 AI Job Market Analytics Dashboard")
st.markdown("Explore hiring trends, most demanded skills, salary distributions, and remote work ratios within the AI industry!")

try:
    with st.spinner("Crunching the job market data..."):
        df = load_data()
        
    # --- Sidebar Filters ---
    st.sidebar.header("Filter Options")
    countries = sorted(df['company_location'].dropna().unique())
    selected_countries = st.sidebar.multiselect("Select Country", options=countries)
    
    levels = sorted(df['experience_level'].dropna().unique())
    selected_levels = st.sidebar.multiselect("Select Experience Level", options=levels)
    
    # Apply Filters
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df['company_location'].isin(selected_countries)]
    if selected_levels:
        filtered_df = filtered_df[filtered_df['experience_level'].isin(selected_levels)]
        
    if filtered_df.empty:
        st.warning("No data available for the applied filters. Please adjust your criteria.")
        st.stop()
        
    st.header("1. Dataset Overview")
    st.dataframe(filtered_df.head(10))

    col1, col2 = st.columns(2)

    with col1:
        st.header("2. Most Demanded Skills")
        skills = filtered_df[filtered_df["required_skills"].notna() & (filtered_df["required_skills"] != "unknown")]["required_skills"]
        vectorizer = CountVectorizer(tokenizer=lambda x: [s.strip() for s in str(x).split(',')], lowercase=True, token_pattern=None)
        
        if not skills.empty:
            skill_matrix = vectorizer.fit_transform(skills)
            skill_counts = pd.DataFrame({
                'skill': vectorizer.get_feature_names_out(),
                'count': skill_matrix.sum(axis=0).A1
            }).sort_values('count', ascending=False).head(15)
            
            fig_skills = px.bar(skill_counts, x='count', y='skill', orientation='h', title="Top 15 Most Demanded Skills", color='count')
            fig_skills.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_skills, use_container_width=True)
        else:
            st.info("Not enough skill data to display.")

    with col2:
        st.header("3. Salary Distribution")
        fig_salary = px.histogram(filtered_df[filtered_df['salary_usd'].notna()], x="salary_usd", nbins=50, title="Salary Distribution (USD)", color_discrete_sequence=['indianred'])
        st.plotly_chart(fig_salary, use_container_width=True)
        
    st.markdown("---")
    st.header("4. Market Trends & Details")
    
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Salary by Experience")
        fig_box = px.box(filtered_df.dropna(subset=['salary_usd', 'experience_level']), x='experience_level', y='salary_usd', title="Salary Distribution across Experience Levels", color='experience_level')
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col4:
        st.subheader("Hiring Timeline")
        timeline_df = filtered_df['posting_date'].dt.to_period('M').value_counts().sort_index().reset_index()
        timeline_df.columns = ['Month', 'Job Count']
        timeline_df['Month'] = timeline_df['Month'].astype(str)
        if not timeline_df.empty:
            fig_timeline = px.line(timeline_df, x='Month', y='Job Count', title='Job Postings Over Time', markers=True)
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Not enough date data to display timeline.")

    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Remote Work Trends")
        df_remote = filtered_df.copy()
        df_remote['remote_ratio'] = df_remote['remote_ratio'].astype(str)
        remote_counts = df_remote['remote_ratio'].replace({'0': '0% (Onsite)', '50': '50% (Hybrid)', '100': '100% (Remote)'}).value_counts().reset_index()
        remote_counts.columns = ['Remote Ratio', 'Count']
        fig_remote = px.pie(remote_counts, values='Count', names='Remote Ratio', title="Remote vs Onsite Jobs", hole=0.4)
        st.plotly_chart(fig_remote, use_container_width=True)

    with col6:
        st.subheader("Skills Word Cloud")
        if not skills.empty and 'skill_counts' in locals():
            wordcloud_freq = dict(zip(skill_counts['skill'], skill_counts['count']))
            if wordcloud_freq:
                wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='viridis').generate_from_frequencies(wordcloud_freq)
                fig_wc, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
        else:
            st.info("Not enough data to generate word cloud.")
            
    st.markdown("---")
    st.header("5. Skill Gap Analysis & Course Recommendations")
    st.markdown("Select your current skills to see if you are ready for the job market. If not, we will suggest the most in-demand missing skills along with courses to upskill!")

    # Normalize and parse skills safely
    def parse_skills(value):
        if pd.isna(value):
            return []
        raw = str(value).strip().lower()
        if raw in {"", "unknown", "none", "nan"}:
            return []
        return [s.strip() for s in raw.split(",") if s.strip() and s.strip() not in {"unknown", "none", "nan"}]

    all_skills_series = df["required_skills"]
    unique_skills = sorted({skill for cell in all_skills_series for skill in parse_skills(cell)})

    suggested_defaults = ["python", "sql", "machine learning"]
    safe_defaults = [s for s in suggested_defaults if s in unique_skills]

    user_skills = st.multiselect(
        "Select the skills you currently possess:",
        options=unique_skills,
        default=safe_defaults
    )

    if st.button("Evaluate My Skills"):
        with st.spinner("Analyzing your profile..."):
            if not user_skills:
                st.warning("Please select at least one skill to evaluate.")
            else:
                user_skills_set = set([s.strip().lower() for s in user_skills])

                def calc_match(req_skills_str):
                    job_skills = set(parse_skills(req_skills_str))
                    if not job_skills:
                        return 0.0, []
                    overlap = job_skills.intersection(user_skills_set)
                    missing = sorted(list(job_skills - user_skills_set))
                    return len(overlap) / len(job_skills), missing

                eval_df = df.copy()

                matches = eval_df["required_skills"].apply(calc_match)
                eval_df["match_score"] = matches.apply(lambda x: x[0])
                eval_df["missing_skills"] = matches.apply(lambda x: x[1])

                top_matches = eval_df.sort_values(by=["match_score", "salary_usd"], ascending=[False, False])
                best_match_score = top_matches.iloc[0]["match_score"] if not top_matches.empty else 0.0

                if best_match_score == 1.0:
                    st.success("🎉 Great news! Your skills perfectly match the requirements for top roles.")
                    st.balloons()
                elif best_match_score >= 0.5:
                    st.info(f"👍 You have a solid foundation! You are up to a {best_match_score*100:.0f}% match for top roles.")
                elif best_match_score > 0:
                    st.warning(f"⚠️ Your maximum match is {best_match_score*100:.0f}%. Upskilling is highly recommended.")
                else:
                    st.error("❌ Your current skills do not closely match the top AI jobs in our database.")

                st.subheader("Jobs You Match Best With:")
                st.dataframe(top_matches[["job_title", "company_location", "salary_usd", "match_score"]].head(5))

                top_potential_jobs = top_matches.head(50)
                all_missing = []
                for missing_list in top_potential_jobs["missing_skills"]:
                    all_missing.extend(missing_list)

                if all_missing:
                    from collections import Counter
                    import urllib.parse

                    most_common_missing = Counter(all_missing).most_common(5)
                    st.subheader("Top Missing Skills Designed Just For You")

                    for skill, count in most_common_missing:
                        encoded_skill = urllib.parse.quote(skill)
                        st.write(f"- 🔸 **{skill.title()}** (Required by {count} of your closest-matching jobs)")
                        st.markdown(f"  *Suggested Courses:* [Coursera](https://www.coursera.org/search?query={encoded_skill})  |  [Udemy](https://www.udemy.com/courses/search/?q={encoded_skill})")
                else:
                    st.write("You don't have any major missing skills for your top matching jobs!")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.warning("Please ensure the ETL pipeline has been run and `data/cleaned_ai_jobs.csv` exists.")
