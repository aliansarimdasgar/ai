from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Define Agents
data_formatter = Agent(
    role="Data Formatting Agent",
    goal="Fetch raw results and convert them into structured format",
    backstory="A data expert that prepares reconciliation data for analysis.",
    llm=llm
)

query_executor = Agent(
    role="Query Execution Agent",
    goal="Run predefined or custom queries on the data to detect patterns",
    backstory="An expert in running and optimizing queries to analyze results.",
    llm=llm
)

insight_generator = Agent(
    role="Insight Generation Agent",
    goal="Analyze discrepancies in data using LLMs and generate insights",
    backstory="A smart AI that identifies hidden patterns and discrepancies.",
    llm=llm
)

summarization_agent = Agent(
    role="Summarization Agent",
    goal="Convert findings into human-readable insights",
    backstory="A language expert that provides summaries for decision-making.",
    llm=llm
)

query_parser = Agent(
    role="Query Parsing Agent",
    goal="Understand user intent and reformulate questions as needed",
    backstory="A chatbot assistant that interprets user queries effectively.",
    llm=llm
)

data_retrieval = Agent(
    role="Data Retrieval Agent",
    goal="Search and retrieve relevant records from the structured dataset",
    backstory="An AI assistant that finds the right data efficiently.",
    llm=llm
)

explanation_agent = Agent(
    role="Explanation Agent",
    goal="Provide a natural language response based on the retrieved data",
    backstory="A helpful AI that explains data findings clearly.",
    llm=llm
)

recommendation_agent = Agent(
    role="Recommendation Agent",
    goal="Suggest corrective actions based on the resolutions",
    backstory="An AI that helps businesses improve based on the analysis.",
    llm=llm
)
