"""
Prompt templates for AI-powered job matching and analysis.

This module contains standardized prompts used by various LLM agents in the system:
- Job post detection and classification
- CV-to-job matching with detailed scoring
- Job post cleaning and standardization
- Structured information extraction
"""

JOB_POST_DETECTION_PROMPT = """Determine if text contains job postings.

Required elements:
- Job title
- Plus at least TWO of:
  - Responsibilities/requirements
  - Application instructions
  - Employment terms (salary, location, type)
  - Company hiring info
  - Recruiter contacts

Not job postings:
- Career advice
- Industry news
- Company updates without hiring
- Educational content
- Networking posts

Return only "True" or "False"."""

SINGLE_JOB_POST_DETECTION_PROMPT = """Determine if text contains EXACTLY ONE job posting.

Single job indicators:
- One job title
- Consistent requirements
- Single set of qualifications

Multiple jobs indicators:
- Multiple job titles
- Different requirements sets
- "Multiple positions" mentions
- Lists of different roles
- Separate sections per position

Return only "True" or "False"."""

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
