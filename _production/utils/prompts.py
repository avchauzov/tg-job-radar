JOB_POST_DETECTION_PROMPT = """You are an expert at analyzing job-related content.
Determine if a text contains ANY job postings.

A job posting MUST include:
    - Specific job title(s)
    - At least TWO of the following:
        - Job responsibilities/requirements
        - Application instructions
        - Employment terms (salary, location, work type)
        - Company hiring information
        - Recruiter or hiring manager contacts

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

CV_MATCHING_PROMPT = """You are a technical recruiter evaluating candidate CVs against job requirements.

Analyze the match between the CV and job post using these three key areas:

1. Experience Match (40% of final score):
   - Years of Experience Match (50%): Compare required vs actual years
   - Domain Knowledge Match (30%): Evaluate industry and domain expertise
   - Project Scale Experience (20%): Assess complexity and scale of projects

2. Skills Match (45% of final score):
   - Technical Skills Match (40%): Must-have technical requirements
   - Education Requirements Match (25%): Required education level
   - Tools and Technologies (20%): Required tools and frameworks
   - Professional Certifications (15%): Relevant certifications

3. Soft Skills Match (15% of final score):
   - Communication Skills (30%): Written and verbal communication abilities
   - Team Collaboration (30%): Experience in team environments
   - Problem-Solving Approach (20%): Analytical and solution-oriented mindset
   - Cultural Values Alignment (20%): Work style and company culture fit

Scoring Guidelines:
95-100: Exceeds all requirements
85-94: Meets all requirements
75-84: Meets most requirements
65-74: Meets basic requirements
50-64: Meets some requirements
0-49: Missing critical requirements

Calculate a single final score (0-100) by:
1. First determining a score for each of the three key areas
2. Then applying the weights: Skills (45%), Experience (40%), Soft Skills (15%)

IMPORTANT: Calculate all numeric values yourself. Do not include any mathematical expressions.
Return only the final calculated score as a single number."""

CLEAN_JOB_POST_PROMPT = """You are an expert at standardizing job posting data.

For each job posting, extract and standardize the following fields:
- job_title: The standardized job title (capitalize words)
- seniority_level: Exactly one of: "Junior", "Mid-Level", "Senior", "Lead", "Principal", "Executive", or empty string if unclear
- location: Format as follows:
    * If full details known: "City, State/Province, Country"
    * If only city and country known: "City, Country"
    * If only country known: "Country"
    * Always capitalize proper nouns
    Never return partial/incomplete locations with commas (e.g., avoid "UK, , ")
- remote_status: Exactly one of: "Remote", "Hybrid", "On-site", or empty string if unclear
- relocation_support: Exactly one of: "Yes", "No", or empty string if not specified
- visa_sponsorship: Exactly one of: "Yes", "No", or empty string if not specified
- salary_range: Standardized format like "$100K-$150K" or empty string if not provided
- company_name: Clean company name (capitalize words)
- description: Cleaned job description

Important rules for standardization:
1. For any missing, undefined, or unclear values, always use an empty string ('') instead of 'N/A', 'None', 'NULL', or similar text
2. Be consistent with capitalization:
   * Use proper capitalization for job titles, company names, and locations
   * Keep "Yes"/"No" responses capitalized
   * Keep status values ("Remote", "Hybrid", "On-site") capitalized
3. Remove any special characters or unnecessary whitespace
4. Ensure values match the expected formats described above
5. If information is ambiguous, prefer empty string over guessing
"""

JOB_POST_PARSING_PROMPT = """You are an expert at parsing job descriptions.
Extract and structure job posting information accurately.

Rules:
    - For any missing or unclear information, use empty string ('')
    - Normalize seniority levels to: Junior, Mid-Level, Senior, Lead, Principal, Executive
    - For remote_status, use only: "Remote", "Hybrid", "On-site", or empty string if unclear
    - Keep the description concise but include all important details and required skills
    - Extract salary range if mentioned, standardize to format like "$100K-$150K"
"""
