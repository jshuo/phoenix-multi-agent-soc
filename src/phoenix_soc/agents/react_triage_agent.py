from typing import Any, Dict, List
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

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

# --- Agent prompt --------------------------------------------------------
SYSTEM = """You are a SOC triage assistant using ReAct. Think step-by-step.
Only call tools when needed. After you finish, output a JSON object with:
- summary, severity (low|med|high), mitre_ttps[], recommendations[]
""".strip()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "Investigate: {task}"),
])

# --- Runner --------------------------------------------------------------
def run_react_triage(task: str, history: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    # In interviews, you can swap model to a local LLM or mock
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    from langchain.agents import AgentExecutor, create_react_agent
    tools = [search_logs, lookup_cti]
    agent = create_react_agent(llm, tools, PROMPT)
    ex = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = ex.invoke({"task": task, "history": history or []})
    return result  # result["output"] contains the JSON string
