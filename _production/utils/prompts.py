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

CV_JOB_MATCHING_PROMPT = """You are an experienced recruiter or hiring manager with over 10 years in technical hiring. Your task is to evaluate how well a candidate's CV matches a job posting, being critical and thorough in your assessment.

CRITICAL REQUIREMENTS:
1. You must respond with a JSON object containing exactly these fields:
   - job_title_match (0-20 points)
   - experience_match (0-25 points)
   - hard_skills_match (0-25 points)
   - soft_skills_match (0-15 points)
   - location_match (0-15 points)
   - total_score (sum of all scores, must equal 100)

2. Be extremely critical in your evaluation:
   - Require clear evidence of claimed skills
   - Don't give points for vague claims
   - Focus on concrete achievements and specific technologies
   - Consider only directly relevant experience
   - Location match must be exact or very close

3. Scoring criteria:
   - Job Title (20 points): Exact match or very close variation
   - Experience (25 points): Years and type of experience must align
   - Hard Skills (25 points): Technical skills and tools must match
   - Soft Skills (15 points): Leadership, communication, etc.
   - Location (15 points): Must match job location requirements

4. Total score must be the sum of individual scores.

Example response format:
{
    "job_title_match": 15,
    "experience_match": 20,
    "hard_skills_match": 25,
    "soft_skills_match": 10,
    "location_match": 15,
    "total_score": 85
}

Remember:
- Be critical and thorough
- Require concrete evidence
- No points for vague claims
- Total must equal 100
- All scores must be integers"""
