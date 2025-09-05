# tests/test_agents.py
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

# Define tools
def search_tool(query: str) -> str:
    # Fake search tool (replace with real search API)
    return f"Search results for '{query}'..."

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for searching information on the web"
    )
]

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are AutoGPT, an autonomous agent with a single goal:

GOAL: {goal}

You can use tools, recall past steps, and create a final answer.
""")

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",  # ReAct-style reasoning
    memory=memory,
    verbose=True
)

# Run AutoGPT-style task
if __name__ == "__main__":
    goal = "explain the concept of autoGPT"
    result = agent.run(prompt.format(goal=goal))
    print("Final Answer:", result)

