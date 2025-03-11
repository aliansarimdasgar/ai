from crewai import Task
from agents import metadata_agent,result_analysis_agent,reporting_agent
from tools import csv_metadata_reader,csv_query_runner


metadata_task = Task(
    description=(
        "Extract relevant metadata from the provided CSV file based on the user query: {query}. "
        "Ensure that only the requested details are returned while maintaining a structured format. "
        "Common metadata elements include column names, row count, column count and file statistics. "
        "If additional clarification is needed, request it before proceeding."
    ),
    agent=metadata_agent,
    expected_output=(
        "A structured JSON or Python dictionary containing the extracted metadata, suitable for further processing.\n\n"
        "**Example Output:**\n"
        "{\n"
        '  "summary": {\n'
        '    "row_count": 1000,\n'
        '    "colun_count": "1000"\n'
        "  },\n"
        '  "columns_name": ["Name","Age","City"]"'
        "}"
    ),
    tools=[csv_metadata_reader]
)

result_analysis_task = Task(
    description=(
        "Analyze the extracted metadata from the previous step and determine if additional data is needed to fully answer the user query: {query}. "
        "If necessary, execute additional queries on the CSV file to extract relevant insights. "
        "Ensure that the output is structured and contains meaningful patterns, trends, anomalies, and any required statistical summaries."
    ),
    agent=result_analysis_agent,
    context=[metadata_task],  # Pass metadata from the first task
    expected_output=(
        "A structured JSON or Python dictionary containing the analyzed insights, ready for final report generation.\n\n"
        "**Example Output:**\n"
        "{\n"
        '  "query": "{query}",\n'
        '  "summary": "The dataset analysis reveals key patterns and trends related to difference in the source and target columns.",\n'
        '  "key_findings": [\n'
        '    {"finding": "Source and target mismathched has happened mostly due to date formating", "data": {"source_date": "18/02/2025", "target_date": 18-02-2025}},\n'
        '    {"finding": "Data is targets are missing but source contains values", "data": {"source_bookId": "BID002", "target_bookId": }}\n'
        "  ],\n"
        '  "data_quality": {\n'
        '    "recommendations": ["Correct the format of the dates in bot source and target"]\n'
        "  },\n"
        "}"
    )
)


reporting_task = Task(
    description=(
        "Generate a structured and visually appealing summary based on the metadata and analysis results, "
        "focusing strictly on the user’s query: {query}. "
        "Ensure the response is concise, relevant, and formatted for readability using bullet points, tables. "
        "Provide insights in an easy-to-understand format, ensuring the output is directly useful to the user."
    ),
    agent=reporting_agent,
    context=[metadata_task, result_analysis_task],  
    expected_output=(
        "A well-structured report answering the user’s query {query}, including:\n"
        "- Direct answers to the query in a concise format.\n"
        # "- **Data Overview**: A structured table displaying relevant information.\n"
        # "- **Trends & Patterns**: Highlights of trends, correlations, or anomalies based on the query.\n"
        # "- **Data Quality Observations**: Any missing data, inconsistencies, or improvements if relevant.\n"
    )
)





# metadata_task = Task(
#     description="Extract and return only the requested metadata from the CSV file based on user query: {query}.",
#     agent=metadata_agent,
#     expected_output="A direct and structured answer to: {query}.",
#     tools=[csv_metadata_reader]
# )

# result_analysis_task = Task(
#     description="Execute user-specified query: {query} on the CSV files to extract relevant insights, trends, or anomalies.",
#     agent=result_analysis_agent,
#     expected_output="A direct and structured response to: {query}.",
#     tools=[csv_query_runner]
# )
# result_analysis_task = Task(
#     description="Analyze the extracted metadata and determine if additional data is needed for the user query {query}."
#                 "If necessary, run queries on the CSV file to extract relevant insights.",
#     agent=result_analysis_agent,
#     context=[metadata_task],  # Pass metadata from the first task
#     expected_output="A structured dataset or insights derived from the user query {query}.",
#     # tools=[csv_query_runner]
# )



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

# reporting_task = Task(
#     description="Summarize the metadata and analysis results relevant to the user’s query in a concise yet complete manner.",
#     agent=reporting_agent,
#     context=[metadata_task, result_analysis_task],  
#     expected_output="A clear and precise summary answering the user's query without unnecessary details."
# )


