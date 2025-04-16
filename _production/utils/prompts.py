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

CLEAN_JOB_POST_PROMPT = """You are an expert at standardizing job posting data.

For each job posting, extract and standardize the following fields:
- job_title: The standardized job title (capitalize words)
- seniority_level: Exactly one of: "Junior", "Mid-Level", "Senior", "Lead", "Principal", "Executive", or empty string if unclear
- location: Format as follows:
    * Physical location + work type: "City, State/Province, Country (Remote)" or "London, UK (Hybrid)" or "Berlin, Germany (On-site)"
    * Remote only: "Remote" or "Remote, US only" or "Remote, EMEA"
    * Always capitalize proper nouns
    Never return partial/incomplete locations
- salary_range: Standardized format like "£100K-150K" or "€100K-150K" (with appropriate currency sign) or empty string if not provided
- company_name: Clean company name (capitalize words)
- description: Cleaned job description containing only essential information:
    * Key responsibilities and requirements
    * Core job duties
    * Critical qualifications
    * Important company context
    * Remove any fluff, marketing language, or redundant information
- skills: List of all technical and professional skills, comma-separated, ordered by importance (most critical first)

Important rules for standardization:
1. For any missing, undefined, or unclear values, always use an empty string ('') instead of 'N/A', 'None', 'NULL', or similar text
2. Be consistent with capitalization:
   * Use proper capitalization for job titles, company names, and locations
   * Keep work type values ("Remote", "Hybrid", "On-site") capitalized
3. Remove any special characters or unnecessary whitespace
4. Ensure values match the expected formats described above
5. If information is ambiguous, prefer empty string over guessing
6. For skills:
   * Extract only technical and professional skills (not soft skills)
   * Order by importance/priority (most critical first)
   * Use standard terminology (e.g., "Python" not "python programming")
   * Remove duplicates
   * Limit to most relevant 15-20 skills total
"""

JOB_POST_PARSING_PROMPT = """You are an expert at parsing job descriptions.
Extract and structure job posting information accurately.

Rules:
    - For any missing or unclear information, use empty string ('')
    - Normalize seniority levels to: Junior, Mid-Level, Senior, Lead, Principal, Executive
    - Format location as "City, Country (Remote/Hybrid/On-site)" or just "Remote" for fully remote positions
    - Clean the description to contain only essential information:
        * Key responsibilities and requirements
        * Core job duties
        * Critical qualifications
        * Important company context
        * Remove any fluff, marketing language, or redundant information
    - Extract salary range if mentioned, standardize to format like "£100K-150K" or "€100K-150K" with appropriate currency sign
    - Extract skills:
        * Combine all technical and professional skills into a single list
        * Order by importance (most critical first)
        * Focus on technical and professional skills only
        * Use standardized terminology
        * Remove duplicates
        * Limit to most relevant 15-20 skills total
"""
