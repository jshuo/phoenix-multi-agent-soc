from typing import Any, Dict, List
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent

# --- Tools ---------------------------------------------------------------
@tool
def search_logs(query: str) -> str:
    """Query SIEM logs. Query examples: 'failed logins last 5m', 'sudo from unknown host'."""
    from ..tools.wazuh_search import mock_search
    return mock_search(query)

@tool
def lookup_cti(indicator: str) -> str:
    """Lookup an IOC/CVE in CTI feeds. Input: ip/domain/hash/CVE."""
    from ..tools.cti_lookup import mock_lookup
    return mock_lookup(indicator)

# --- Policy snippets (lightweight) --------------------------------------
from .safety_policies import action_is_safe

# --- ReAct Prompt Template -----------------------------------------------
REACT_PROMPT = PromptTemplate.from_template("""You are a SOC triage assistant. Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer as JSON with summary, severity (low|med|high), mitre_ttps[], recommendations[]

Begin!

Question: {input}
Thought: {agent_scratchpad}""")

# --- Runner --------------------------------------------------------------
def run_react_triage(task: str, history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    # In interviews, you can swap model to a local LLM or mock
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    tools = [search_logs, lookup_cti]
    
    # Use the proper ReAct template
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=10,  # Prevent infinite loops
        handle_parsing_errors=True
    )
    
    result = executor.invoke({"input": f"Investigate: {task}"})
    return result

# Alternative: Using LangGraph ReAct (more modern approach)
def run_langgraph_react_triage(task: str) -> Dict[str, Any]:
    """Modern LangGraph-based ReAct implementation"""
    from langgraph.prebuilt import create_react_agent
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_logs, lookup_cti]
    
    # LangGraph handles ReAct pattern internally
    agent_executor = create_react_agent(llm, tools)
    
    result = agent_executor.invoke({
        "messages": [("human", f"Investigate this SOC incident: {task}. "
                              "Provide final answer as JSON with summary, severity, mitre_ttps, recommendations.")]
    })
    
    return result