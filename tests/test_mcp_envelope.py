from src.phoenix_soc.audit.mcp_envelope import build_envelope
from src.phoenix_soc.tools.pkcs11_signer import sign_payload

def test_build_and_sign():
    env = build_envelope("p", "m", {"a":1}, {"b":2})
    j = env.to_json()
    sig = sign_payload(j)
    assert isinstance(sig, str) and len(sig) == 64
