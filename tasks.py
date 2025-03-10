from crewai import Task
from agents import metadata_agent,result_analysis_agent,reporting_agent
from tools import csv_metadata_reader,csv_query_runner

metadata_task = Task(
    description="Extract and return only the requested metadata from the CSV file based on user query: {query}.",
    agent=metadata_agent,
    expected_output="A direct and structured answer to: {query}.",
    tools=[csv_metadata_reader]
)

# result_analysis_task = Task(
#     description="Execute user-specified query: {query} on the CSV files to extract relevant insights, trends, or anomalies.",
#     agent=result_analysis_agent,
#     expected_output="A direct and structured response to: {query}.",
#     tools=[csv_query_runner]
# )
result_analysis_task = Task(
    description="Analyze the extracted metadata and determine if additional data is needed for the user query {query}."
                "If necessary, run queries on the CSV file to extract relevant insights.",
    agent=result_analysis_agent,
    context=[metadata_task],  # Pass metadata from the first task
    expected_output="A structured dataset or insights derived from the user query {query}.",
    # tools=[csv_query_runner]
)



# reporting_task = Task(
#     description="Generate a structured report based on metadata and analysis results. "
#                 "Format the report as requested (e.g., tabular, bullet points, or plain text).",
#     agent=reporting_agent,
#     context=[metadata_task, result_analysis_task],  
#     expected_output="A well-structured report in the requested format, summarizing key insights clearly.",
# )

# reporting_task = Task(
#     description="Generate a concise, one-line summary of metadata and analysis results.",
#     agent=reporting_agent,
#     context=[metadata_task, result_analysis_task],  
#     expected_output="A direct answer in a single line.",
# )

reporting_task = Task(
    description="Summarize the metadata and analysis results relevant to the user’s query in a concise yet complete manner.",
    agent=reporting_agent,
    context=[metadata_task, result_analysis_task],  
    expected_output="A clear and precise summary answering the user's query without unnecessary details."
)


