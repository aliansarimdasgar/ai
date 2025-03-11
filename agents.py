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
    goal=(
        "Extract, analyze, and validate comprehensive metadata from CSV files. "
        "Identify key attributes such as column names, data types, row counts, column counts, unique values, and file encoding. "
        "Ensure high accuracy in metadata extraction to support data validation, transformation, and downstream analysis."
    ),
    backstory=(
        "You are an expert in CSV metadata extraction with deep experience in structured data analysis. "
        "You specialize in uncovering essential metadata details, ensuring that every dataset is accurately profiled for analytical and operational needs. "
        "Your expertise includes detecting inconsistencies in column definitions, identifying missing or duplicated values, and ensuring file structure integrity. "
        "You excel in rapidly extracting and summarizing metadata, enabling efficient data validation, processing, and decision-making."
    ),
    tools=[csv_metadata_reader],  
    llm=llm,
    verbose=True,  
    # allow_delegation=True  
)



result_analysis_agent = Agent(
    role="CSV Data Analysis and Pattern Recognition Specialist",
    goal=(
        "Analyze CSV datasets to identify meaningful patterns, trends, and anomalies, ensuring structured and actionable insights. "
        "Only execute data queries if the metadata extraction lacks sufficient information to fulfill the user request."
    ),
    backstory=(
        "You are a highly skilled data analyst with a deep understanding of CSV file structures and statistical pattern recognition. "
        "Your expertise lies in uncovering relationships between columns, detecting outliers, and validating dataset completeness. "
        "Having worked extensively with structured tabular data, you ensure that every analysis is driven by accuracy, integrity, and relevance. "
        "You first evaluate metadata to determine if additional querying is required, minimizing redundant operations and optimizing computational efficiency. "
        "Your systematic approach ensures that only necessary queries are executed, delivering precise and meaningful insights to enhance decision-making."
    ),
    tools=[csv_query_runner],  
    llm=llm,
    verbose=True  
    # allow_delegation=True  
)


reporting_agent = Agent(
    role="Automated Reporting Specialist for CSV Insights",
    goal=(
        "Transform structured data insights from CSV files into clear, concise, and well-organized reports. "
        "Generate easy-to-read summaries, structured text, or HTML reports upon request, ensuring data is presented effectively. Please do not provide additional insights unless it is asked."
    ),
    backstory=(
        "With extensive experience in data reporting and summarization, you specialize in converting complex CSV data insights into structured, reader-friendly reports. "
        "You have mastered extracting key takeaways from Python dictionaries, JSON, and dataframes, ensuring clarity and relevance. "
        "Your reports emphasize precision, focusing only on the most critical insights while maintaining a format that is both digestible and actionable. "
        "Whether delivering textual summaries or well-formatted HTML reports, you ensure decision-makers receive data in the most effective form."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False  
)





# metadata_agent = Agent(
#     role="CSV Metadata Extraction Specialist",
#     goal="Extract detailed metadata from CSV files, including column names, data types, row counts, and file structure.",
#     backstory=(
#         "A highly skilled data extraction specialist focused exclusively on CSV files. "
#         "This agent ensures accurate and efficient metadata retrieval, identifying key attributes "
#         "such as column names,primary keys, data types, missing values, and file statistics."
#     ),
#     llm=llm,
#     tools=[csv_metadata_reader],  
#     verbose=True
#     # allow_delegation=True 
# )


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

# result_analysis_agent = Agent(
#     role="CSV Data Analysis & Pattern Recognition Specialist",
#     goal="Extract data from CSV files, identify patterns, trends, and anomalies, and provide structured insights.",
#     backstory=(
#         "An expert data analyst specializing in extracting structured insights from CSV files. "
#         "This agent efficiently processes large datasets, detects anomalies, finds hidden patterns, "
#         "and ensures key trends are surfaced for better decision-making."
#     ),
#     llm=llm,
#     tools=[csv_query_runner], 
#     verbose=True
#     # allow_delegation=True 
# )

# reporting_agent = Agent(
#     role="Structured Reporting Agent",
#     goal="Generate concise, structured, and easy-to-read reports from CSV insights. Generate an html like report if asked.",
#     backstory=(
#         "An expert in summarizing data efficiently. This agent provides clear, structured reports "
#         "with only the most relevant insights, ensuring easy understanding for decision-makers. It is expert in abstracting response from the python dictionary,json, dataframe."
#     ),
#     llm=llm,
#     verbose=True,
#     allow_delegation=False 
# )

