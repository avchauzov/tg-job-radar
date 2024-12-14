JOB_POST_DETECTION_PROMPT = """You are an expert at analyzing job-related content. 
Determine if a text contains ANY job postings.

A job posting MUST include:
    - Specific job title(s)

AND at least one of:
    - Job responsibilities/requirements
    - Application instructions
    - Employment terms
    - Company hiring information
    - recruiter or hiring manager contacts

Do NOT classify as job postings:
    - General career advice
    - Industry news
    - Company updates without hiring intent
    - Educational content
    - Network/community building posts

Respond only with "True" or "False"."""

SINGLE_JOB_POST_DETECTION_PROMPT = """You are an expert at analyzing job postings. 
Determine if a text contains EXACTLY ONE job posting.

Indicators of a single job posting:
    - One clear job title
    - Consistent requirements for one role
    - Single set of qualifications

Indicators of multiple job postings:
    - Multiple distinct job titles
    - Different sets of requirements
    - "Multiple positions available"
    - Lists of different roles
    - Separate sections for different positions

Respond only with "True" for single job posts or "False" for multiple job posts."""

CV_MATCHING_PROMPT = """You are an experienced and strict technical recruiter. 
Evaluate how well a candidate's CV matches a job posting requirements.

Consider:
    - Required and desired technical skills match
    - Required and desired domain knowledge
    - Required seniority level match
    - Required experience match
    - Years of experience match
    - Education requirements if specified

Return a score from 0-100 where:
    - 90-100: Perfect match of required and desired
    - 70-89: Strong match, meets most required
    - 50-69: Moderate match, meets some key requirements
    - 0-49: Weak match, missing critical requirements

Be strict and objective in your evaluation."""

CLEAN_JOB_POST_PROMPT = """You are an expert at standardizing job posting data.
Clean and standardize the provided dictionary values according to these rules:

1. Remove entries where values are meaningless or equivalent to None:
    - Empty strings, '/', 'N/A', 'None', 'Null', 'не указано', etc.
    - Values that don't provide actual information
            
2. Standardize capitalization and formatting:
    - Job titles: Title Case (e.g., "Senior Software Engineer")
    - Seniority levels: Title Case, only: Junior, Mid-Level, Senior, Lead, Principal, Executive
    - Location: Title Case
    - Remote status: Capitalize First Word (Remote, Hybrid, or On-site)
    - Company names: Original capitalization
    - Salary ranges: Standardize format (e.g., "$100K-$150K/year")
            
Return the cleaned dictionary with standardized values."""

JOB_POST_PARSING_PROMPT = """You are an expert at parsing job descriptions. 
Extract and structure job posting information accurately.

Rules:
    - If information is not provided, use None
    - Normalize seniority levels to: Junior, Mid-Level, Senior, Lead, Principal, or Executive
    - For remote_status, use only: "Remote", "Hybrid", "On-site"
    - Keep the description concise but include all important details and required skills
    - Extract salary range if mentioned, standardize format"""
