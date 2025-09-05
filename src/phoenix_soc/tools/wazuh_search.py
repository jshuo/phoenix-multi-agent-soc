import json, random

def mock_search(query: str) -> str:
    # return a tiny JSON so tools stay deterministic during interviews
    sample = {
        "query": query,
        "hits": [
            {"host": "fab-gw-01", "event": "failed_login", "ip": "203.0.113.10", "count": random.randint(3,9)},
            {"host": "mes-app-02", "event": "sudo_from_unknown_host", "ip": "198.51.100.7", "count": 1},
        ],
        "window": "5m"
    }
    return json.dumps(sample)
