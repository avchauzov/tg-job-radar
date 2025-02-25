import json
import logging
from typing import Dict, List

from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic

from _production import LLM_BASE_MODEL, MATCH_SCORE_THRESHOLD, STAGING_DATA__POSTS
from _production.utils.common import setup_logging
from _production.utils.sql import fetch_from_db

setup_logging(__file__[:-3])
logger = logging.getLogger(__name__)


def get_job_posts_with_scores() -> List[Dict]:
    """
    Fetch job posts with their relevance scores from the database
    """
    columns = ["post_structured", "score"]
    _, data = fetch_from_db(
        table=STAGING_DATA__POSTS,
        select_condition=", ".join(columns),  # limit last
    )

    return [dict(zip(columns, values)) for values in data]


def analyze_token_effectiveness(
    posts: List[Dict], current_tokens: List[str]
) -> Dict[str, float]:
    """
    Analyze how effective each token is in identifying relevant job posts
    Returns a dictionary with precision and occurrence rate for each token
    """
    token_stats = {
        token: {"true_positives": 0, "false_positives": 0, "total_occurrences": 0}
        for token in current_tokens
    }

    total_posts = len(posts)

    for post in posts:
        post_text = json.dumps(post["post_structured"]).lower()
        is_relevant = post["score"] >= MATCH_SCORE_THRESHOLD

        for token in current_tokens:
            if token.lower() in post_text:
                token_stats[token]["total_occurrences"] += 1
                if is_relevant:
                    token_stats[token]["true_positives"] += 1
                else:
                    token_stats[token]["false_positives"] += 1

    # Calculate precision and occurrence rate for each token
    token_metrics = {}
    for token, stats in token_stats.items():
        total_matches = stats["true_positives"] + stats["false_positives"]
        occurrence_rate = (
            stats["total_occurrences"] / total_posts if total_posts > 0 else 0
        )

        precision = stats["true_positives"] / total_matches if total_matches > 0 else 0
        token_metrics[token] = {
            "precision": precision,
            "occurrence_rate": occurrence_rate,
        }

    return token_metrics


def create_token_optimization_agent():
    """
    Create a LangChain agent for optimizing prefiltering tokens using Anthropic's Claude
    """
    llm = ChatAnthropic(
        model_name=LLM_BASE_MODEL, temperature=0.3, timeout=120, stop=None
    )

    # Get posts from database or other source
    posts = get_job_posts_with_scores()

    tools = [
        Tool(
            name="analyze_token_effectiveness",
            func=lambda x: analyze_token_effectiveness(
                posts=posts, current_tokens=x.split(", ")
            ),
            description="Analyzes how effective each token is in identifying relevant job posts. Input should be comma-separated tokens.",
        ),
        Tool(
            name="suggest_new_tokens",
            func=lambda x: suggest_new_tokens(posts=posts),
            description="Analyzes the job posts content and suggests new potential tokens that might be effective for filtering.",
        ),
    ]

    from langchain.agents import AgentType, initialize_agent

    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True
    )

    return agent


def optimize_prefiltering_tokens(current_tokens: List[str]) -> List[str]:
    """
    Optimize the prefiltering tokens by suggesting additional tokens
    Returns a list of optimized tokens
    """
    try:
        agent = create_token_optimization_agent()
        posts = get_job_posts_with_scores()

        # Step 1: Analyze current tokens' effectiveness
        metrics = analyze_token_effectiveness(posts, current_tokens)

        # Step 2: Get suggestions for new tokens based on current ones
        prompt = f"""
        Based on these current tokens: {current_tokens},
        suggest 5-10 additional tokens that might be effective.
        Focus on technical terms, job titles, and skills.

        Rules:
        - Return ONLY a comma-separated list of tokens
        - Do not number the items
        - Do not include explanations
        - Do not include tokens that are already in the current list

        Example response format:
        python, tensorflow, deep learning, pytorch, spark
        """

        new_suggestions = agent.invoke({"input": prompt, "chat_history": []})

        if isinstance(new_suggestions, dict) and "output" in new_suggestions:
            # Split the comma-separated response and clean each token
            suggested_tokens = [
                token.strip().lower()
                for token in new_suggestions["output"].split(",")
                if token.strip() and token.strip().lower() not in current_tokens
            ]

            # Combine current tokens with new suggestions, removing duplicates
            optimized_tokens = list(set(current_tokens + suggested_tokens))

            # Log the metrics for monitoring
            logging.info(f"Current tokens: {current_tokens}")
            logging.info(f"New suggestions: {suggested_tokens}")
            logging.info(f"Combined tokens: {optimized_tokens}")

            return optimized_tokens

        logging.error(f"Unexpected response format from agent: {new_suggestions}")
        return current_tokens

    except Exception as error:
        logging.error(f"Error optimizing tokens: {str(error)}")
        return current_tokens


def suggest_new_tokens(posts: List[Dict]) -> Dict[str, float]:
    """
    Analyze job posts content and suggest new potential tokens
    Returns a dictionary of suggested tokens with their potential effectiveness scores
    """
    relevant_posts = [post for post in posts if post["score"] >= MATCH_SCORE_THRESHOLD]

    # Combine all relevant post content
    all_text = " ".join(
        [json.dumps(post["post_structured"]).lower() for post in relevant_posts]
    )

    # Extract common phrases and words that appear frequently in relevant posts
    suggested_tokens = {}

    # Use the LLM to analyze the content and suggest tokens
    llm = ChatAnthropic(model_name=LLM_BASE_MODEL, temperature=0, timeout=120)

    prompt = f"""
    Analyze this job posts content and suggest 5-10 specific tokens (words or short phrases)
    that might be effective for identifying relevant job posts. Focus on technical terms,
    job titles, and key skills. Return the response as a simple comma-separated list of tokens.

    Content to analyze: {all_text[:2000]}  # Limiting content length for API
    """

    try:
        response = llm.invoke(prompt)
        new_tokens = [token.strip() for token in response.content.split(",")]

        # Analyze effectiveness of suggested tokens
        token_metrics = analyze_token_effectiveness(posts, new_tokens)

        return token_metrics
    except Exception as e:
        logger.error(f"Error suggesting new tokens: {str(e)}")
        return {}
