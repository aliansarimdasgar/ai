from crewai import Crew, Process
from agents import metadata_agent, result_analysis_agent, reporting_agent
from tasks import metadata_task, result_analysis_task, reporting_task

crew = Crew(
    agents=[metadata_agent, result_analysis_agent, reporting_agent],
    tasks=[metadata_task, result_analysis_task, reporting_task],
    process=Process.sequential, 
    allow_delegation=True ,
    # memory=True,
    # verbose=True
)
