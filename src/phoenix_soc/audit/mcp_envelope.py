import json, hashlib, time
from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class Envelope:
    prompt: str
    model: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    timestamp: float
    model_hash: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    signature: str | None = None  # produced by HSM/PKCS#11

    def finalize(self):
        self.input_hash  = hashlib.sha256(json.dumps(self.inputs,  sort_keys=True).encode()).hexdigest()
        self.output_hash = hashlib.sha256(json.dumps(self.outputs, sort_keys=True).encode()).hexdigest()
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

def build_envelope(prompt: str, model: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Envelope:
    env = Envelope(prompt=prompt, model=model, inputs=inputs, outputs=outputs, timestamp=time.time())
    return env.finalize()
