from dotenv import load_dotenv
load_dotenv()
import queue
import threading
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Optional if you prefer messages-style:
# from langchain_core.messages import HumanMessage

log_queue = queue.Queue()
enriched_queue = queue.Queue()

# --- Build the LLM chain once (thread-safe to call) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # ensure OPENAI_API_KEY is set
prompt = ChatPromptTemplate.from_template(
    "Classify this log: {log}\n"
    "Reply with exactly one line: [NORMAL] or [ALERT] + a short reason."
)
chain = prompt | llm | StrOutputParser()

def log_producer():
    logs = [
        "User admin failed login from IP 10.0.0.5",
        "Database connection timeout",
        "PUF-HSM signed log: integrity OK",
        "Suspicious sudo attempt from unknown host",
    ]
    for log in logs:
        print(f"[Producer] New log -> {log}")
        log_queue.put(log)
        time.sleep(0.5)
    # graceful shutdown signal
    log_queue.put(None)

def analyzer_agent():
    while True:
        log = log_queue.get()
        if log is None:
            enriched_queue.put(None)
            log_queue.task_done()
            break
        try:
            # ✅ Correct v0.2 usage
            result = chain.invoke({"log": log})
            enriched = f"{log} --> {result.strip()}"
            print(f"[Analyzer] {enriched}")
            enriched_queue.put(enriched)
        except Exception as e:
            print(f"[Analyzer][ERROR] {e!r} on log: {log}")
        finally:
            log_queue.task_done()

def siem_agent():
    while True:
        enriched = enriched_queue.get()
        if enriched is None:
            enriched_queue.task_done()
            break
        # TODO: POST to OpenSearch/Wazuh here
        print(f"[SIEM] store: {enriched}")
        enriched_queue.task_done()

# Start threads
threads = [
    threading.Thread(target=log_producer, daemon=True),
    threading.Thread(target=analyzer_agent, daemon=True),
    threading.Thread(target=siem_agent, daemon=True),
]
for t in threads:
    t.start()

# Wait for queues to drain (optional in tests)
log_queue.join()
enriched_queue.join()
