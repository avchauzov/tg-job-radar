"""
Prompt templates for AI-powered job matching and analysis.

This module contains standardized prompts used by various LLM agents in the system:
- Job post detection and classification
- CV-to-job matching with detailed scoring
- Job post cleaning and standardization
- Structured information extraction
"""

JOB_POST_DETECTION_PROMPT = """You are an expert at detecting job postings in text. Your task is to determine if the given text contains a job posting.

A valid job posting MUST have:
1. A specific job title (e.g., "Senior Software Engineer", "Data Scientist")
2. At least TWO of the following:
   - Job responsibilities or requirements
   - Application instructions or process
   - Employment terms (salary, location, work type)
   - Company hiring information
   - Recruiter or hiring manager contacts

Examples of job postings:
✅ "We're hiring a Senior Python Developer! Requirements: 5+ years Python, Django, FastAPI. Location: Berlin. Salary: €80K-100K. Send CV to jobs@company.com"
✅ "Looking for a Data Scientist to join our team. Must have ML experience and Python skills. Remote work possible. Competitive salary."
✅ "Join our team as a Frontend Developer! React, TypeScript required. Full-time position in Munich. Benefits include health insurance."

Examples of non-job postings:
❌ "Tips for writing a good CV"
❌ "Our company just launched a new product"
❌ "Join our networking event next week"
❌ "Learn Python in 30 days"
❌ "Looking for a co-founder for my startup"

IMPORTANT: If you are unsure or hesitate whether the text is a job posting, return "True". It's better to process a potential job posting than to miss one.

Return only "True" if the text is a job posting, "False" otherwise."""

SINGLE_JOB_POST_DETECTION_PROMPT = """You are an expert at detecting single job postings in text. Your task is to determine if the given text contains EXACTLY ONE job posting.

A single job posting MUST have:
1. Exactly one specific job title
2. One consistent set of requirements
3. One set of qualifications
4. One location/type of work
5. One application process

Examples of single job postings:
✅ "We're hiring a Senior Python Developer! Requirements: 5+ years Python, Django, FastAPI. Location: Berlin. Salary: €80K-100K. Send CV to jobs@company.com"
✅ "Looking for a Data Scientist to join our team. Must have ML experience and Python skills. Remote work possible. Competitive salary."

Examples of multiple job postings:
❌ "We're hiring for multiple positions: 1) Senior Python Developer 2) Junior Frontend Developer 3) DevOps Engineer"
❌ "Open positions: - Backend Developer - Frontend Developer - QA Engineer"
❌ "Join our team! We have openings for: * Software Engineer * Data Scientist * Product Manager"

Examples of non-job postings:
❌ "Tips for writing a good CV"
❌ "Our company just launched a new product"
❌ "Join our networking event next week"

IMPORTANT: If you are unsure or hesitate whether the text contains exactly one job posting, return "True". It's better to process a potential single job posting than to miss one.

Return only "True" if the text contains exactly one job posting, "False" otherwise."""

CV_JOB_MATCHING_PROMPT = """You are an expert CV analyzer. Evaluate the match between a CV and a job posting.

Score each category separately and independently:
- Hard skills (40 points): technical skills, programming languages, frameworks, tools, architecture patterns
- Experience (40 points): relevant experience, similar projects, domain knowledge, task scale
- Soft skills and seniority (20 points): communication, leadership, team work, independence

IMPORTANT:
- Never assign the same score for all cases.
- Always analyze the actual content and provide a realistic, differentiated score for each category.
- Do not always return average or maximum scores. Use the full range based on the match quality.

Scoring guidelines:
Hard skills (40 points):
- 35-40: All required skills + additional relevant skills
- 25-34: All required skills
- 15-24: Most required skills
- 0-14: Missing critical skills

Experience (40 points):
- 35-40: Extensive relevant experience + similar projects
- 25-34: Good relevant experience
- 15-24: Some relevant experience
- 0-14: Limited relevant experience

Soft skills and seniority (20 points):
- 15-20: Perfect match for seniority level + strong soft skills
- 10-14: Good match for seniority level
- 5-9: Some relevant soft skills
- 0-4: Limited soft skills match

Examples:
1. Excellent match:
{
    "hard_skills": 39,
    "experience": 38,
    "soft_skills": 18
}
2. Good technical, weak soft skills:
{
    "hard_skills": 32,
    "experience": 30,
    "soft_skills": 7
}
3. Lacking experience:
{
    "hard_skills": 28,
    "experience": 12,
    "soft_skills": 15
}
4. Poor match:
{
    "hard_skills": 10,
    "experience": 8,
    "soft_skills": 3
}

Return only the scores in this exact JSON format:
{
    "hard_skills": <score 0-40>,
    "experience": <score 0-40>,
    "soft_skills": <score 0-20>
}
"""

CLEAN_JOB_POST_PROMPT = """You are an expert at standardizing and parsing job postings.

For each job posting, extract and standardize these fields:
- job_title: Standardized title (capitalize words)
- seniority_level: One of: "Junior", "Mid-Level", "Senior", "Lead", "Principal", "Executive", or empty string
- location: Format as "City, Country (Remote/Hybrid/On-site)" or "Remote" for fully remote
- salary_range: Format like "€100K-150K" with appropriate currency sign, or empty string
- company_name: Clean company name (capitalize words)
- skills: Comma-separated list of technical/professional skills, ordered by importance (15-20 max)
- short_description: Concise 1-2 sentence summary of the role and key responsibilities

Standardization rules:
1. For any missing, undefined, or unclear values, always use an empty string ('') instead of 'N/A', 'None', 'NULL', or similar text
2. Capitalize proper nouns and work types
3. Remove special characters and extra whitespace
4. For skills:
   - Focus on technical/professional skills only
   - Use standard terminology
   - Remove duplicates
   - Order by importance
   - Limit to most relevant 4-8 skills
5. For short_description:
   - Keep it concise (1-2 sentences)
   - Focus on key responsibilities and role purpose
   - Avoid generic phrases
   - Include main technologies or domains if relevant

Return only the standardized fields in a structured format."""

JOB_POST_REWRITE_PROMPT = """You are an expert at rewriting job postings to be clear and concise while preserving all essential information.

Your task is to rewrite the job posting to be more structured and focused, removing any unnecessary content while maintaining technical accuracy.

Guidelines:
1. Keep all technical requirements and skills with exact versions/levels
2. Keep all mandatory qualifications with specific years of experience
3. Keep salary and location information
4. Keep application process details
5. Maintain technical accuracy:
   - Preserve exact technology names and versions
   - Keep specific architecture patterns
   - Maintain precise skill requirements
   - Preserve exact certification requirements
6. Remove:
   - Marketing fluff and buzzwords
   - Generic company descriptions
   - Redundant information
   - Unnecessary formatting
   - Emojis and special characters
   - Multiple spaces and line breaks

IMPORTANT: The rewritten job posting must be at least 128 characters long.

Return your response in this exact JSON format:
{
    "summary": "<your rewritten job posting here>"
}

The rewritten job posting should be clear, concise, and maintain all essential technical information."""

CV_SUMMARY_PROMPT = """You are a professional CV analyzer. Create concise summaries while preserving technical details.

Focus on:
1. Technical skills and expertise
2. Key professional achievements
3. Relevant work experience
4. Education and certifications

IMPORTANT: The summary must be at least 128 characters long.

Return your response in this exact JSON format:
{
    "summary": "<your concise summary here>"
}

The summary should maintain a professional tone and technical accuracy."""
