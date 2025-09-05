# tests/test_agents.py
from dotenv import load_dotenv
load_dotenv()
# autogpt_demo.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

def build_memory():
    # Empty vector store to start; it will grow as the agent saves context.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_texts(["AutoGPT scratchpad initialized."], embedding=embeddings)
    return vs.as_retriever(search_kwargs={"k": 4})

def build_tools():
    ddg = DuckDuckGoSearchRun()
    return [
        Tool(
            name="web-search",
            func=ddg.run,  # or ddg.invoke in LC 0.3+
            description="Search the web for up-to-date facts and URLs."
        ),
    ]

def build_llm():
    # You can swap this for any LangChain ChatModel
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def main():
    llm = build_llm()
    tools = build_tools()
    memory = build_memory()

    # Create AutoGPT from LLM and tools (convenience constructor)
    agent = AutoGPT.from_llm_and_tools(
        ai_name="ResearchBuddy",
        ai_role="An autonomous research assistant that finds facts and summarizes them.",
        tools=tools,
        llm=llm,
        memory=memory,                 # VectorStoreRetriever
        # feedback_tool=None,          # Optional: Human-in-the-loop confirmation tool
        # chat_history_memory=None,    # Optional: a chat history store
    )

    # Goals are a list of tasks the agent should autonomously pursue
    goals = [
    "Analyze security incidents in fab logs",
    "Summarize anomalies and mitigation steps in one sentence"
    ]

    # Run AutoGPT (it plans, calls tools, writes to memory, and produces a final answer)
    result = agent.run(goals)  # returns a string (final report/summary)
    print("\n=== FINAL ANSWER ===\n", result)

if __name__ == "__main__":
    main()
