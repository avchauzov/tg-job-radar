import json
import logging
from typing import Dict, Optional

from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic

from _production import LLM_BASE_MODEL
from _production.utils.exceptions import LLMError


def parse_llm_response(response: str | list) -> Dict:
    """Safely parse LLM response into a dictionary"""
    try:
        # Convert response to string if it's a list
        json_str = response[0] if isinstance(response, list) else response
        json_str = str(json_str).strip()

        # Find the first '{' and last '}' to extract JSON
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}") + 1

        if start_idx != -1 and end_idx != -1:
            json_str = json_str[start_idx:end_idx]

            # Remove any markdown code block markers
            json_str = json_str.replace("```json", "").replace("```", "")

            return json.loads(json_str)
        else:
            logging.error(f"No JSON object found in response: {response}")
            return {"score": 0, "reasoning": "No valid JSON found in response"}

    except Exception as error:
        logging.error(f"Failed to parse LLM response: {response}")
        return {"score": 0, "reasoning": f"Error parsing response: {str(error)}"}


def create_cv_matching_agent(cv_content: str):
    """Create a LangChain agent for CV matching"""
    llm = ChatAnthropic(
        model_name=LLM_BASE_MODEL, temperature=0.0, timeout=120, stop=["```"]
    )

    def analyze_experience_match(job_post: str) -> Dict[str, float]:
        """Analyze experience match between CV and job post"""
        try:
            response = llm.invoke(
                f"""You are a technical recruiter evaluating candidate CVs against job requirements.

                Analyze the match focusing on these weighted criteria:
                - Years of Experience Match (50%): Compare required vs actual years
                - Domain Knowledge Match (30%): Evaluate industry and domain expertise
                - Project Scale Experience (20%): Assess complexity and scale of projects

                Score Guidelines:
                95-100: Exceeds all requirements
                85-94: Meets all requirements
                75-84: Meets most requirements
                65-74: Meets basic requirements
                50-64: Meets some requirements
                0-49: Missing critical requirements

                Provide your response in this exact JSON format:
                {{
                    "score": <number between 0-100>
                }}

                CV: {cv_content}

                Job Post: {job_post}"""
            )
            return parse_llm_response(response.content)
        except Exception as error:
            raise LLMError(f"Experience analysis failed: {str(error)}")

    def analyze_skills_match(job_post: str) -> Dict[str, float]:
        """Analyze technical skills match between CV and job post"""
        try:
            response = llm.invoke(
                f"""You are a technical recruiter evaluating candidate CVs against job requirements.

                Analyze the match focusing on these weighted criteria:
                - Technical Skills Match (40%): Must-have technical requirements
                - Education Requirements Match (25%): Required education level
                - Tools and Technologies (20%): Required tools and frameworks
                - Professional Certifications (15%): Relevant certifications

                Score Guidelines:
                95-100: Exceeds all requirements
                85-94: Meets all requirements
                75-84: Meets most requirements
                65-74: Meets basic requirements
                50-64: Meets some requirements
                0-49: Missing critical requirements

                Provide your response in this exact JSON format:
                {{
                    "score": <number between 0-100>
                }}

                CV: {cv_content}

                Job Post: {job_post}"""
            )
            return parse_llm_response(response.content)
        except Exception as error:
            raise LLMError(f"Skills analysis failed: {str(error)}")

    def analyze_soft_skills_match(job_post: str) -> Dict[str, float]:
        """Analyze soft skills and cultural fit"""
        try:
            response = llm.invoke(
                f"""You are a technical recruiter evaluating candidate CVs against job requirements.

                Analyze the soft skills and cultural fit between CV and job post. Score based on these criteria:
                - Communication Skills (30%): Written and verbal communication abilities
                - Team Collaboration (30%): Experience in team environments
                - Problem-Solving Approach (20%): Analytical and solution-oriented mindset
                - Cultural Values Alignment (20%): Work style and company culture fit

                Provide your response in this exact JSON format:
                {{
                    "score": <number between 0-100>
                }}

                CV: {cv_content}

                Job Post: {job_post}"""
            )
            return parse_llm_response(response.content)
        except Exception as error:
            raise LLMError(f"Soft skills analysis failed: {str(error)}")

    tools = [
        Tool(
            name="analyze_experience_match",
            func=analyze_experience_match,
            description="Analyzes experience match between CV and job post",
        ),
        Tool(
            name="analyze_skills_match",
            func=analyze_skills_match,
            description="Analyzes technical skills match between CV and job post",
        ),
        Tool(
            name="analyze_soft_skills_match",
            func=analyze_soft_skills_match,
            description="Analyzes soft skills and cultural fit",
        ),
    ]

    return initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True
    )


def enhanced_cv_matching(cv_content: str, job_post: str) -> Optional[float]:
    """Enhanced CV matching using LangChain agent"""
    try:
        agent = create_cv_matching_agent(cv_content)

        result = agent.invoke(
            {
                "input": f"""Analyze this job post for compatibility with the CV.
            Use the tools to evaluate matches, then calculate final score with these weights:
            - Skills Match (45%): Technical skills, education, tools, certifications
            - Experience Match (40%): Years, domain knowledge, project scale
            - Soft Skills Match (15%): Communication, teamwork, culture fit

            Score Guidelines:
            95-100: Exceeds all primary and secondary requirements
            85-94: Meets all primary and most secondary requirements
            75-84: Meets primary requirements with some secondary gaps
            65-74: Meets most primary requirements with significant secondary gaps
            50-64: Meets some primary requirements
            0-49: Missing critical primary requirements

            Provide your final response as a JSON object with this structure:
            {{
                "experience_match": {{ "score": number}},
                "skills_match": {{ "score": number}},
                "soft_skills_match": {{ "score": number}}
            }}

            Job Post: {job_post}""",
                "chat_history": [],
            }
        )

        if isinstance(result, dict) and "output" in result:
            analysis = result["output"]

            # Extract scores
            experience_score = float(analysis["experience_match"]["score"])
            skills_score = float(analysis["skills_match"]["score"])
            soft_skills_score = float(analysis["soft_skills_match"]["score"])

            # Calculate weighted final score
            final_score = int(
                (skills_score * 0.45)
                + (experience_score * 0.40)
                + (soft_skills_score * 0.15)
            )

            return final_score

        return None

    except Exception as error:
        logging.error(f"Enhanced CV matching failed: {str(error)}")
        return None
