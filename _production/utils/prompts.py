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

EXPERIENCE_MATCHING_PROMPT = """Evaluate experience match between CV and job requirements.

Areas to assess:
- Years of relevant experience
- Domain knowledge and expertise
- Project scale and complexity
- Career progression

Return only a single integer value:
1 - Exceeds or meets all requirements
2 - Meets most requirements
3 - Misses critical requirements"""

SKILLS_MATCHING_PROMPT = """Evaluate skills match between CV and job requirements.

Areas to assess:
- Technical skills and proficiency
- Education and qualifications
- Tools and technologies familiarity
- Professional certifications

Return only a single integer value:
1 - Exceeds or meets all requirements
2 - Meets most requirements
3 - Misses critical requirements"""

SOFT_SKILLS_MATCHING_PROMPT = """Evaluate soft skills match between CV and job requirements.

Areas to assess:
- Communication abilities
- Team collaboration experience
- Problem-solving approach
- Cultural fit and adaptability

Return only a single integer value:
1 - Exceeds or meets all requirements
2 - Meets most requirements
3 - Misses critical requirements"""

CLEAN_JOB_POST_PROMPT = """You are an expert at standardizing and parsing job postings.

For each job posting, extract and standardize these fields:
- job_title: Standardized title (capitalize words)
- seniority_level: One of: "Junior", "Mid-Level", "Senior", "Lead", "Principal", "Executive", or empty string
- location: Format as "City, Country (Remote/Hybrid/On-site)" or "Remote" for fully remote
- salary_range: Format like "€100K-150K" with appropriate currency sign, or empty string
- company_name: Clean company name (capitalize words)
- skills: Comma-separated list of technical/professional skills, ordered by importance (15-20 max)

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

Return only the standardized fields in a structured format."""

JOB_POST_REWRITE_PROMPT = """You are an expert at rewriting job postings to be clear and concise while preserving all essential information.

Your task is to rewrite the job posting to be more structured and focused, removing any unnecessary content.

Guidelines:
1. Keep all technical requirements and skills
2. Keep all mandatory qualifications
3. Keep salary and location information
4. Keep application process details
5. Remove:
   - Marketing fluff and buzzwords
   - Generic company descriptions
   - Redundant information
   - Unnecessary formatting
   - Emojis and special characters
   - Multiple spaces and line breaks

Structure the output as:
1. Job Title
2. Key Requirements
3. Technical Skills
4. Location & Type
5. Salary (if available)
6. Application Process

Example input:
"🚀 Join our amazing team! We're looking for a Senior Python Developer to help us build the future!
Our company is a leading tech innovator with a great culture and work-life balance.
Requirements:
- 5+ years of Python experience
- Django, FastAPI
- PostgreSQL, Redis
- Docker, Kubernetes
Location: Berlin (Hybrid)
Salary: €80K-100K
Send your CV to jobs@company.com"

Example output:
"Senior Python Developer

Key Requirements:
- 5+ years of Python development experience
- Strong expertise in web development
- Experience with microservices architecture

Technical Skills:
- Python, Django, FastAPI
- PostgreSQL, Redis
- Docker, Kubernetes
- AWS (ECS, Lambda)

Location: Berlin (Hybrid)
Salary: €80K-100K

To apply, send your CV to jobs@company.com"

Return only the rewritten job posting text."""
