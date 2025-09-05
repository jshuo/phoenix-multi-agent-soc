import json

def mock_lookup(indicator: str) -> str:
    db = {
        "203.0.113.10": {"threat": "LockBit infra", "confidence": 0.82},
        "198.51.100.7": {"threat": "benign", "confidence": 0.12},
        "CVE-2024-12345": {"threat": "RCE", "cvss": 9.8},
    }
    return json.dumps(db.get(indicator, {"threat": "unknown"}))
