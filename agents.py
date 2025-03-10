from crewai import Agent, LLM
import os
from tools import csv_metadata_reader, csv_query_runner
from dotenv import load_dotenv

load_dotenv()

# Get API Key
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("API Key not found! Make sure your .env file is set up correctly.")

# Define LLM Model
llm = LLM(
    model="mistral/mistral-small-latest",
    temperature=0.5,
    api_key=api_key
)


metadata_agent = Agent(
    role="CSV Metadata Extraction Specialist",
    goal="Extract detailed metadata from CSV files, including column names, data types, row counts, and file structure.",
    backstory=(
        "A highly skilled data extraction specialist focused exclusively on CSV files. "
        "This agent ensures accurate and efficient metadata retrieval, identifying key attributes "
        "such as column names,primary keys, data types, missing values, and file statistics."
    ),
    llm=llm,
    tools=[csv_metadata_reader],  
    verbose=True
    # allow_delegation=True 
)


# result_analysis_agent = Agent(
#     role="Result Analysis Agent",
#     goal="Analyze the data from the result CSV files and run predefined queries.",
#     backstory="A seasoned data analyst with expertise in extracting insights from structured datasets."
#               "This agent specializes in running queries on CSV files, identifying patterns, trends, and anomalies, "
#               "and providing actionable insights for decision-making. With strong analytical skills and an eye for "
#               "detail, the agent ensures that all relevant data points are examined efficiently.",
#     llm=llm,
#     tools=[csv_query_runner],
#     allow_delegation=True
# )

result_analysis_agent = Agent(
    role="CSV Data Analysis & Pattern Recognition Specialist",
    goal="Extract data from CSV files, identify patterns, trends, and anomalies, and provide structured insights.",
    backstory=(
        "An expert data analyst specializing in extracting structured insights from CSV files. "
        "This agent efficiently processes large datasets, detects anomalies, finds hidden patterns, "
        "and ensures key trends are surfaced for better decision-making."
    ),
    llm=llm,
    tools=[csv_query_runner], 
    verbose=True
    # allow_delegation=True 
)

reporting_agent = Agent(
    role="Structured Reporting Agent",
    goal="Generate concise, structured, and easy-to-read reports from CSV insights. Generate an html like report if asked.",
    backstory=(
        "An expert in summarizing data efficiently. This agent provides clear, structured reports "
        "with only the most relevant insights, ensuring easy understanding for decision-makers. It is expert in abstracting response from the python dictionary,json, dataframe."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False 
)

