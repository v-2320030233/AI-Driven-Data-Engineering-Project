-- SQL Analytics Queries for AI Job Market

-- Note: These queries assume you are using PostgreSQL and have loaded the 
-- `cleaned_ai_jobs.csv` into a table named `cleaned_ai_jobs`.

-- 1. Top demanded skills
-- (Assuming skills are comma-separated strings natively in PG)
SELECT skill, COUNT(*) as demand
FROM (
    SELECT unnest(string_to_array(required_skills, ',')) as skill
    FROM cleaned_ai_jobs
) AS skills
WHERE skill != 'unknown' AND skill IS NOT NULL
GROUP BY skill
ORDER BY demand DESC
LIMIT 15;

-- 2. Average salary by job title
SELECT job_title, AVG(salary_usd) as avg_salary
FROM cleaned_ai_jobs
WHERE salary_usd IS NOT NULL
GROUP BY job_title
ORDER BY avg_salary DESC;

-- 3. Job demand by country (company_location)
SELECT company_location as country, COUNT(*) as job_count
FROM cleaned_ai_jobs
GROUP BY company_location
ORDER BY job_count DESC;

-- 4. Remote vs onsite job distribution
SELECT remote_ratio, COUNT(*) as job_count
FROM cleaned_ai_jobs
GROUP BY remote_ratio
ORDER BY remote_ratio;

-- 5. Hiring trends by industry (grouping by month/year)
SELECT industry, DATE_TRUNC('month', posting_date) as hiring_month, COUNT(*) as job_count
FROM cleaned_ai_jobs
WHERE posting_date IS NOT NULL
GROUP BY industry, hiring_month
ORDER BY industry, hiring_month;
