# Minimal stub – replace with real PKCS#11 calls in production
import hashlib

def sign_payload(data: str, key_ref: str = "demo-key-001") -> str:
    # demonstration: pretend the HSM signs by hashing with a key ref salt
    return hashlib.sha256((key_ref + data).encode()).hexdigest()
