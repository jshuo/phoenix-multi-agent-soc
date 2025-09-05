from src.phoenix_soc.audit.mcp_envelope import build_envelope
from src.phoenix_soc.tools.pkcs11_signer import sign_payload
from src.phoenix_soc.audit.verifier import verify_signature

if __name__ == "__main__":
    env = build_envelope("demo", "local-llm", {"x":1}, {"y":2})
    env.signature = sign_payload(env.to_json())
    j = env.to_json()
    print(j)
    print("Verified:", verify_signature(j))
