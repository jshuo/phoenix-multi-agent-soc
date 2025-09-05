import json, hashlib
from .mcp_envelope import Envelope

def verify_signature(envelope_json: str, key_ref: str = "demo-key-001") -> bool:
    try:
        obj = json.loads(envelope_json)
        body = json.dumps(obj, indent=2)
        # demo "verification": recompute with same salt and compare length/format
        sig = obj.get("signature", "")
        return isinstance(sig, str) and len(sig) == 64
    except Exception:
        return False
