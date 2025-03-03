from crewai import Task
from agents import (
    data_formatter, query_executor, insight_generator, summarization_agent,
    query_parser, data_retrieval, explanation_agent, recommendation_agent
)

# Define tasks for each agent
data_formatting_task = Task(
    description="Fetch raw reconciliation results and convert them into a structured format.",
    agent=data_formatter
)

query_execution_task = Task(
    description="Run queries on the dataset to identify mismatches and patterns.",
    agent=query_executor,
    context=[data_formatting_task]
)

insight_generation_task = Task(
    description="Analyze discrepancies and generate insights based on the data.",
    agent=insight_generator,
    context=[query_execution_task]
)

summarization_task = Task(
    description="Summarize the key findings and provide actionable insights.",
    agent=summarization_agent,
    context=[insight_generation_task]
)

query_parsing_task = Task(
    description="Understand user queries and reformat them for structured data retrieval.",
    agent=query_parser
)

data_retrieval_task = Task(
    description="Search for relevant records from the structured dataset.",
    agent=data_retrieval,
    context=[query_parsing_task]
)

explanation_task = Task(
    description="Provide clear responses based on retrieved data insights.",
    agent=explanation_agent,
    context=[data_retrieval_task]
)

recommendation_task = Task(
    description="Suggest corrective actions based on analysis and insights.",
    agent=recommendation_agent,
    context=[explanation_task]
)

# List of all tasks
all_tasks = [
    data_formatting_task, query_execution_task, insight_generation_task,
    summarization_task, query_parsing_task, data_retrieval_task,
    explanation_task, recommendation_task
]
