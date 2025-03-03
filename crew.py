from crewai import Crew
from agents import (
    data_formatter, query_executor, insight_generator, summarization_agent,
    query_parser, data_retrieval, explanation_agent, recommendation_agent
)
from tasks import all_tasks

# Define Crew
crew = Crew(
    agents=[
        data_formatter, query_executor, insight_generator, summarization_agent,
        query_parser, data_retrieval, explanation_agent, recommendation_agent
    ],
    tasks=all_tasks
)
